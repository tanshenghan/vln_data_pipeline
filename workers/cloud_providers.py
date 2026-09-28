"""Small OpenAPI adapters for Volcengine ML 2024-07-01 and Baidu AIHC v2.

Only create idle jobs, list/inspect jobs and stop a recorded job ID. See the
official API/SDK references in docs/CLOUD_COLLECTION.md for wire contracts.
"""
import hashlib
import hmac
import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit

import httpx


class CloudError(RuntimeError):
    pass


def mac(key, value):
    return hmac.new(key, value.encode(), hashlib.sha256).digest()


def query_string(query):
    return "&".join(f"{quote(str(k), safe='-_.~')}={quote(str(v), safe='-_.~')}" for k, v in sorted(query.items()))


def signed_headers(pool, action, query, body, *, timestamp=None, metadata=False):
    ak, sk, token = pool.credentials()
    timestamp = timestamp or datetime.now(timezone.utc)
    headers = {"Host": urlsplit(pool.endpoint).netloc, "Content-Type": "application/json"}
    if pool.provider == "volcengine":
        date = timestamp.strftime("%Y%m%dT%H%M%SZ")
        headers.update({"X-Date": date, "X-Content-Sha256": hashlib.sha256(body).hexdigest()})
        if token:
            headers["X-Security-Token"] = token
        normalized = {k.lower(): v for k, v in headers.items()}
        names = ";".join(sorted(normalized))
        canonical = "\n".join(["POST", "/", query_string(query),
                               "".join(f"{k}:{normalized[k]}\n" for k in sorted(normalized)),
                               names, headers["X-Content-Sha256"]])
        scope = f"{date[:8]}/{pool.region}/ml_platform/request"
        key = sk.encode()
        for part in (date[:8], pool.region, "ml_platform", "request"):
            key = mac(key, part)
        signature = mac(key, "\n".join(["HMAC-SHA256", date, scope, hashlib.sha256(canonical.encode()).hexdigest()])).hex()
        headers["Authorization"] = f"HMAC-SHA256 Credential={ak}/{scope}, SignedHeaders={names}, Signature={signature}"
    else:
        headers["version" if metadata else "X-API-Version"] = "v2"
        if token:
            headers["x-bce-security-token"] = token
        date = timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        prefix = f"bce-auth-v1/{ak}/{date}/1800"
        key = mac(sk.encode(), prefix).hex().encode()
        normalized = {k.lower(): v for k, v in headers.items()}
        names = (["host"] if metadata else
                 sorted(k for k in normalized if k in {"host", "content-type"} or k.startswith("x-bce-")))
        canonical = "\n".join(["GET" if metadata else "POST", "/", query_string(query),
            "\n".join(f"{quote(k, safe='-_.~')}:{quote(normalized[k].strip(), safe='-_.~')}" for k in names)])
        headers["Authorization"] = f"{prefix}/{';'.join(names)}/{mac(key, canonical).hex()}"
    return headers


@dataclass(frozen=True)
class Job:
    id: str
    name: str
    state: str
    reason: str = ""
    idle: bool = False

    @property
    def terminal(self):
        return self.state in {"succeeded", "failed", "stopped"}

    @property
    def preempted(self):
        return any(word in self.reason.lower() for word in ("preempt", "reclaim", "抢占", "闲时资源回收"))


def normalize_job(raw, provider):
    if provider == "volcengine":
        status = raw["Status"]
        state = status["State"]
        reason = " ".join(str(v or "") for v in (raw.get("StopReason"), status.get("Message"),
                                                  status.get("RetryTrigger"), status.get("SecondaryState")))
        identity, name = raw["Id"], raw["Name"]
        idle = raw.get("ResourceConfig", {}).get("Preemptible") is True
    else:
        state = raw["status"]
        reason = " ".join(str(raw.get(k) or "") for k in ("reason", "message", "statusReason"))
        # AIHC exposes idle-resource reclamation on the current instance, not
        # necessarily on the job. Historical preemptions must not hide a new
        # application error after a provider-managed restart of the same job.
        reason += " " + " ".join(str(pod.get(k) or "") for pod in (raw.get("pods") or [])
                                 for k in ("status", "podPhase", "reason", "message"))
        identity, name = raw["jobId"], raw["name"]
        idle = raw.get("idleTask") is True or raw.get("priority") == "lowest"
    state = state.lower().replace("_", "").replace("-", "")
    if state in {"pending", "queued", "queuing", "queueing", "inqueue", "created", "creating", "submitted",
                 "scheduling", "scheduled", "deploying", "starting", "initializing", "waiting", "restarting", "retrying", "abnormal"}:
        state = "pending"
    elif state in {"succeeded", "succeed", "success", "completed", "complete"}:
        state = "succeeded"
    elif state in {"stopped", "cancelled", "canceled", "terminated", "killed", "manualtermination"}:
        state = "stopped"
    elif state in {"failed", "error"}:
        state = "failed"
    elif state in {"stopping", "terminating", "killing"}:
        state = "stopping"
    elif state != "running":
        raise CloudError(f"未知云任务状态 {state}，停止派发以免误判为空闲")
    return Job(str(identity), name, state, reason, idle)


class Provider:
    def __init__(self, pool, client=None):
        self.pool = pool
        self.client = client or httpx.Client(timeout=15, follow_redirects=False)

    def close(self):
        self.client.close()

    def call(self, action, payload, *, queue=None):
        pool = self.pool
        if pool.provider == "volcengine":
            query = {"Action": action, "Version": "2024-07-01"}
        else:
            query = {"action": action, "resourcePoolId": pool.resource_pool_id}
            if action != "DescribeJobs":
                query["queueID"] = queue or pool.queue_id
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        return self._request(action, query, body)

    def baidu_metadata(self, action, **query):
        # Resource/development APIs use GET and a different v2 header/signature
        # from training APIs. Do not add the training-only aihc-serverless route.
        if self.pool.provider != "baidu" or action not in {
                "DescribeResourcePools", "DescribeResourcePool", "DescribeQueues",
                "DescribeQueue", "DescribeDevInstances"}:
            raise ValueError("不支持的百度只读资源查询")
        return self._request(action, {"action": action, **query}, b"", metadata=True)

    def _request(self, action, query, body, *, metadata=False):
        pool = self.pool
        headers = signed_headers(pool, action, query, body, metadata=metadata)
        try:
            response = self.client.request("GET" if metadata else "POST",
                pool.endpoint.rstrip("/") + "/?" + query_string(query), content=body, headers=headers)
            if response.status_code >= 300:
                raise CloudError(f"{pool.name} {action}: HTTP {response.status_code}")
            result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # No request headers, signed URL or provider bodies in logs/UI.
            raise CloudError(f"{pool.name} {action}: 请求未确认（{type(exc).__name__}）") from None
        error = result.get("ResponseMetadata", {}).get("Error")
        if error or result.get("code"):
            raise CloudError(f"{pool.name} {action}: 云 API 拒绝请求，请在云审计中核对")
        return result.get("Result", result)

    def list_jobs(self):
        result = {}
        for queue in self.pool.monitor_queue_ids or [self.pool.queue_id]:
            queue_jobs = {}
            for page in range(1, 1001):
                if self.pool.provider == "volcengine":
                    data = self.call("ListJobs", {"ResourceQueueId": queue, "PageNumber": page, "PageSize": 100})
                    items, total = data["Items"], int(data["TotalCount"])
                else:
                    data = self.call("DescribeJobs", {"queue": queue, "pageNumber": page, "pageSize": 100})
                    items, total = data["jobs"], int(data["totalCount"])
                for raw in items:
                    job = normalize_job(raw, self.pool.provider)
                    queue_jobs[job.id] = job
                if page * 100 >= total:
                    break
                if not items:
                    raise CloudError("云任务分页结果不完整，停止派发")
            else:
                raise CloudError("云任务分页超过上限，停止派发")
            if self.pool.provider == "baidu":
                self._baidu_other_workloads(queue, queue_jobs)
            else:
                self._volc_other_workloads(queue, queue_jobs)
            result.update(queue_jobs)
        return list(result.values())

    def _volc_other_workloads(self, queue, jobs):
        # Development instances and inference deployments compete with training
        # jobs. Inservice deployments can still be waiting for scale-out replicas.
        specs = [("ListDevInstances", "List", "dev", {"ResourceQueueIds": [queue]}),
                 ("ListDeployments", "Items", "deployment", {"ResourceQueueId": queue})]
        for action, field, prefix, filters in specs:
            seen = set()
            for page in range(1, 1001):
                data = self.call(action, {**filters, "PageNumber": page, "PageSize": 100})
                items, total = data[field], int(data["TotalCount"])
                for raw in items:
                    if raw.get("ResourceQueueId", queue) != queue:
                        raise CloudError("火山负载队列筛选结果不匹配，停止派发")
                    state = raw["Status"]["State"]
                    if state.lower() == "inservice":
                        roles = raw.get("RoleStatus")
                        if not isinstance(roles, list) or not roles:
                            raise CloudError("火山服务缺少实例数量，停止派发")
                        state = "Running"
                        for role in roles:
                            counts = role["Status"]
                            for expected, available in (("ExpectedReplicias", "AvailableReplicas"),
                                                        ("ExpectedUpdatedReplicas", "UpdatedAvailableReplicas")):
                                if int(counts[expected]) > int(counts[available]):
                                    state = "Pending"
                    elif state.lower() in {"deploying", "updating", "scaling"}:
                        state = "Pending"
                    identity = f"{prefix}:{raw['Id']}"
                    # Synthetic IDs are only monitored, never passed to StopJob.
                    jobs[identity] = normalize_job({"Id": identity, "Name": identity,
                                                   "Status": {"State": state}}, "volcengine")
                    seen.add(raw["Id"])
                if page * 100 >= total:
                    if len(seen) < total:
                        raise CloudError("火山负载分页结果不完整，停止派发")
                    break
                if not items:
                    raise CloudError("火山负载分页结果不完整，停止派发")
            else:
                raise CloudError("火山负载分页超过上限，停止派发")

    def _baidu_other_workloads(self, queue, jobs):
        # Waiting development instances also need GPUs; a training-only list
        # is insufficient. These synthetic IDs can never be stopped by us.
        states = {0: "pending", 1: "pending", 2: "pending", 3: "running",
                  4: "stopping", 5: "stopped", 6: "failed", 7: "stopping", 8: "stopped"}
        for page in range(1, 1001):
            data = self.baidu_metadata("DescribeDevInstances", queueName=queue, onlyMyDevs="false",
                                       pageNumber=page, pageSize=100)
            items, total = data["devInstances"], int(data["totalCount"])
            for raw in items:
                if raw["queueName"] != queue:
                    raise CloudError("百度开发机队列筛选结果不匹配，停止派发")
                state = states.get(raw["status"])
                if state is None:
                    raise CloudError("未知百度开发机状态，停止派发")
                identity = f"dev:{raw['id']}"
                jobs[identity] = Job(identity, identity, state)
            if page * 100 >= total:
                break
            if not items:
                raise CloudError("百度开发机分页结果不完整，停止派发")
        else:
            raise CloudError("百度开发机分页超过上限，停止派发")
        detail = self.baidu_metadata("DescribeQueue", queueId=queue)
        # AIHC can return the physical parent even when queried with a child
        # ID; use only the exact matching elastic child, never a sibling.
        matches = []
        pending = [detail]
        while pending:
            item = pending.pop()
            if item.get("queueId") == queue and item.get("queueType") == "Elastic":
                matches.append(item)
            pending.extend(item.get("children", []))
        if len(matches) != 1:
            raise CloudError("采集必须使用百度弹性任务队列 ID，不能使用物理队列或资源池 ID")
        detail = matches[0]
        # Include queue-wide waiting workloads absent from training/dev APIs
        # (e.g. services). Subtract known pending jobs, including our idle jobs,
        # so a collector waiting for its GPU does not preempt itself.
        waiting = int(detail.get("inqueueJobs", 0))
        if not 0 <= waiting <= 10000:
            raise CloudError("百度队列等待任务数量无效，停止派发")
        extra = max(0, waiting - sum(job.state == "pending" for job in jobs.values()))
        for index in range(extra):
            identity = f"queue:{queue}:waiting:{index}"
            jobs[identity] = Job(identity, identity, "pending")

    def get(self, job_id):
        if self.pool.provider == "volcengine":
            raw = self.call("GetJob", {"Id": job_id})
        else:
            raw = self.call("DescribeJob", {"jobId": job_id, "needDetail": True})
            raw = raw.get("job", raw)
        return normalize_job(raw, self.pool.provider)

    def payload(self, run_id, generation, job_name, *, authenticate_image=True):
        pool = self.pool
        args = ["env", *(f"{k}={v}" for k, v in sorted(pool.worker_env.items())), pool.remote_python,
                "-m", "workers.cloud_entry", "--run-id", run_id, "--generation", str(generation),
                "--storage-root", str(pool.remote_storage_root), "--storage-id", pool.storage_id,
                "--grace-seconds", str(pool.stop_grace_seconds)]
        log_path = pool.remote_storage_root / "runs" / run_id / f"cloud_entry-g{generation}.log"
        command = (f"umask 077; cd {shlex.quote(str(pool.remote_repo))} && exec {shlex.join(args)}"
                   f" >> {shlex.quote(str(log_path))} 2>&1")
        if pool.run_as_user:
            uid, gid = pool.run_as_user.split(":")
            command = shlex.join(["exec", "setpriv", f"--reuid={uid}", f"--regid={gid}",
                                  "--clear-groups", "/bin/sh", "-c", command])
        payload = pool.template()
        if pool.provider == "volcengine":
            payload.update(Name=job_name, ProjectName=pool.project_name)
            resource = payload.setdefault("ResourceConfig", {})
            roles = resource.get("Roles", [])
            if len(roles) != 1 or roles[0].get("Replicas") != 1:
                raise ValueError("采集模板必须为单实例任务")
            resource.update(ResourceQueueId=pool.queue_id, Preemptible=True, Priority=1,
                            ResourceReservation=False, HoldingTimeSeconds=0)
            resource.pop("ResourceReservationPlanId", None)
            payload.setdefault("RuntimeConfig", {})["Command"] = command
            # Retries belong to our durable controller. No hidden successor jobs.
            payload["RetryConfig"] = {"MaxRetryTimes": 0}
        else:
            if payload.get("jobSpec", {}).get("replicas") != 1:
                raise ValueError("采集模板必须为单实例任务")
            payload.update(name=job_name, command=command, jobType="PyTorchJob", idleTask=True, faultTolerance=False)
            payload.pop("priority", None)  # idleTask fixes priority=lowest; specifying both is rejected.
            payload.pop("faultToleranceArgs", None)
            payload.pop("faultToleranceConfig", None)
            if pool.baidu_registry_id:
                allowed = {f"{pool.baidu_registry_id}-{network}.cnc.{pool.region}.baidubce.com"
                           for network in ("vpc", "pub")}
                if payload["jobSpec"].get("image", "").split("/", 1)[0] not in allowed:
                    raise ValueError("采集镜像地址与配置的 CCR 实例不匹配")
                if authenticate_image:
                    payload["jobSpec"]["imageConfig"] = {
                        "username": pool.baidu_registry_username, "password": self._registry_password()}
        return payload

    def _registry_password(self):
        pool = self.pool
        ak, sk, token = pool.credentials()
        host = f"ccr.{pool.region}.baidubce.com"
        path = f"/v1/instances/{pool.baidu_registry_id}/credential"
        date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        headers = {"host": host, "x-bce-date": date}
        if token:
            headers["x-bce-security-token"] = token
        prefix = f"bce-auth-v1/{ak}/{date}/1800"
        key = mac(sk.encode(), prefix).hex().encode()
        names = sorted(headers)
        canonical = "\n".join(["POST", path, "",
            "\n".join(f"{quote(k, safe='-_.~')}:{quote(headers[k], safe='-_.~')}" for k in names)])
        headers["authorization"] = f"{prefix}/{';'.join(names)}/{mac(key, canonical).hex()}"
        try:
            response = self.client.post(f"https://{host}{path}", headers=headers, json={"duration": 1})
            if response.status_code != 200:
                raise CloudError(f"{pool.name} 无法取得镜像临时凭据：HTTP {response.status_code}")
            password = response.json().get("password")
            if not isinstance(password, str) or not password:
                raise CloudError("CCR 未返回有效镜像临时凭据")
            return password
        except (httpx.HTTPError, ValueError):
            raise CloudError("CCR 镜像凭据请求未确认，暂不提交云任务") from None

    def submit(self, payload):
        result = self.call("CreateJob", payload)
        job_id = result.get("Id") if self.pool.provider == "volcengine" else result.get("jobId")
        if not job_id:
            raise CloudError("提交响应缺少任务 ID，等待核对，禁止重复创建")
        return str(job_id)

    def stop(self, job_id):
        if self.pool.provider == "volcengine":
            self.call("StopJob", {"Id": job_id, "Reason": "VLN idle collection yielding resources"})
        else:
            self.call("StopJob", {"jobId": job_id})
