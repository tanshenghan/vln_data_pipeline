from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


class RunRequest(BaseModel):
    kind: Literal["gt", "seedance"] = "gt"
    name: str = Field(default="", max_length=120)
    dataset: Literal["r2r", "rxr", "scalevln", "srdf", "vlnverse"] = "r2r"
    split: Literal["train", "val_seen", "val_unseen", "test"] = "train"
    role: Literal["guide", "follower", "coarse", "fine"] = "guide"
    languages: list[Literal["all", "en-US", "en-IN", "hi-IN", "te-IN"]] = ["all"]
    count: int = Field(default=1, ge=1, le=1000)
    episode_ids: list[str] = Field(default_factory=list, max_length=1000)
    source_ids: list[str] = Field(default_factory=list, max_length=1000)
    full_dataset: bool = False
    routes_per_shard: int = Field(default=8, ge=1, le=64)
    workers: int = Field(default=1, ge=1, le=2)
    execution: Literal["local", "cloud_idle"] = "local"
    compute_pool: str = Field(default="auto", pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    duration: int = Field(default=15, ge=4, le=15)
    lightnav_max_steps: int = Field(default=500, ge=1, le=1000)
    instruction: str = Field(default="", max_length=4000)
    operation: Literal["generate", "preview"] = "generate"
    preview_id: UUID | None = None

    @model_validator(mode="after")
    def check(self):
        if self.dataset == "srdf":
            # A visible dataset option is not permission to treat graph nodes as GT.
            from data_pipeline.srdf_source import COLLECTION_REASON
            raise ValueError(COLLECTION_REASON)
        if self.dataset == "scalevln":
            if self.kind != "gt" or self.split != "train" or self.role != "guide" or self.languages != ["all"]:
                raise ValueError("ScaleVLN 连续轨迹仅支持 GT、train、guide、all")
            if self.full_dataset:
                raise ValueError("ScaleVLN 请按已部署场景选择数量或 episode；当前不支持全量分片")
            if self.execution != "local":
                raise ValueError("ScaleVLN 当前接入本机采集；云环境需另行部署 HM3D 与源数据")
        if self.execution == "cloud_idle" and self.kind != "gt":
            raise ValueError("云端闲时调度目前用于 GT 采集")
        if self.execution == "local" and self.compute_pool != "auto":
            raise ValueError("本机任务不指定云资源池")
        if self.kind != "seedance" and (self.operation == "preview" or self.preview_id):
            raise ValueError("Prompt 预览仅适用于 Seedance")
        if self.kind == "seedance" and self.full_dataset:
            raise ValueError("Seedance 请显式指定数量或源样本，不支持全量 GT 采集选项")
        if self.operation == "preview" and self.preview_id:
            raise ValueError("预览任务不能引用另一个预览")
        if self.dataset == "r2r" and (self.role != "guide" or self.languages != ["all"]):
            raise ValueError("R2R 使用 guide 和 all；语言筛选仅用于 RxR")
        if self.dataset == "vlnverse":
            if self.kind != "gt":
                raise ValueError("VLNVerse 当前接入场景浏览与 GT 采集")
            if self.role not in {"coarse", "fine"} or self.languages != ["all"]:
                raise ValueError("VLNVerse 使用 coarse/fine 和 all")
            if self.full_dataset:
                raise ValueError("VLNVerse 请按已部署场景选择 episode，暂不支持全量采集")
        elif self.role not in {"guide", "follower"}:
            raise ValueError("R2R/RxR 使用 guide/follower 角色")
        if self.split == "test" and self.dataset != "vlnverse":
            raise ValueError("R2R/RxR test 未公开专家轨迹")
        if not self.languages or ("all" in self.languages and len(self.languages) != 1):
            raise ValueError("语言必须为 all 或明确语言列表")
        if self.full_dataset and (self.episode_ids or self.source_ids):
            raise ValueError("全量任务不能同时指定 episode")
        if len(set(self.episode_ids)) != len(self.episode_ids):
            raise ValueError("episode ID 不得重复")
        return self


class CollectionPlanRequest(RunRequest):
    execution: Literal["local", "cloud_idle", "mixed"] = "cloud_idle"
    compute_pools: list[str] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def check_plan(self):
        if self.dataset not in {"r2r", "rxr", "scalevln"} or self.kind != "gt":
            raise ValueError("持续采集支持 R2R / RxR / ScaleVLN GT")
        if self.execution == "local" and self.compute_pools != ["local"]:
            raise ValueError("本机持续采集只能选择 GPU0（local）")
        if self.execution == "cloud_idle" and "local" in self.compute_pools:
            raise ValueError("同时选择本机与云端时使用 mixed 持续采集计划")
        if self.execution == "mixed" and ("local" not in self.compute_pools or len(self.compute_pools) < 2):
            raise ValueError("混合持续采集需要同时选择 GPU0 和至少一个云资源池")
        if len(set(self.compute_pools)) != len(self.compute_pools):
            raise ValueError("采集设备不得重复")
        return self


class ReviewRequest(BaseModel):
    decision: Literal["keep", "exclude", "review"]
    note: str = Field(default="", max_length=4000)
