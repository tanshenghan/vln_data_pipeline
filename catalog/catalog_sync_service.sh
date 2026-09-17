#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-status}"
PROJECT_ROOT="/mnt/pfs/wangqiongyan/BetaWXM0.1"
PYTHON_BIN="/mnt/pfs/wangqiongyan/.conda/envs/starVLA/bin/python"
SYNC_SCRIPT="${PROJECT_ROOT}/scripts/dataset_semantic_explorer/sync_catalog.py"
DATASET_ROOT="/mnt/cfs/data/datasets"
CATALOG_PATH="/mnt/pfs/wangqiongyan/dataset_semantic_explorer/catalog.json"
INSTRUCTION_CLASSIFICATION_PATH="/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_classification.checkpoint.jsonl"
INSTRUCTION_INVENTORY_PATH="/mnt/pfs/wangqiongyan/dataset_semantic_explorer/instruction_inventory.jsonl"
PFS_CONVERSION_ROOT="/mnt/pfs/data/datasets/lerobotv3/high_priority_80d"
CFS_CONVERSION_ROOT="/mnt/cfs/data/datasets/lerobotv3/high_priority_80d"
RUNTIME_ROOT="/mnt/pfs/wangqiongyan/dataset_semantic_explorer"
STATUS_PATH="${RUNTIME_ROOT}/catalog_sync_status.json"
LOCK_PATH="${RUNTIME_ROOT}/catalog_sync.lock"
PID_PATH="${RUNTIME_ROOT}/catalog_sync.pid"
LOG_PATH="${RUNTIME_ROOT}/catalog_sync.log"
INTERVAL_SECONDS=3600

sync_arguments=(
  --root "${DATASET_ROOT}"
  --catalog "${CATALOG_PATH}"
  --instruction-inventory "${INSTRUCTION_INVENTORY_PATH}"
  --instruction-classification "${INSTRUCTION_CLASSIFICATION_PATH}"
  --conversion-pfs-root "${PFS_CONVERSION_ROOT}"
  --conversion-cfs-root "${CFS_CONVERSION_ROOT}"
  --status-path "${STATUS_PATH}"
  --lock-path "${LOCK_PATH}"
  --minimum-records 1000
  --max-unknown-increase 0
)

running_pid() {
  local candidate=""
  if [[ ! -f "${PID_PATH}" ]]; then
    return 1
  fi
  read -r candidate < "${PID_PATH}" || return 1
  if [[ ! "${candidate}" =~ ^[0-9]+$ ]] || ! kill -0 "${candidate}" 2>/dev/null; then
    return 1
  fi
  if [[ ! -r "/proc/${candidate}/cmdline" ]] || ! tr '\0' ' ' < "/proc/${candidate}/cmdline" | grep -Fq "${SYNC_SCRIPT} watch"; then
    return 1
  fi
  printf '%s\n' "${candidate}"
}

start_service() {
  mkdir -p "${RUNTIME_ROOT}"
  local existing=""
  if existing="$(running_pid)"; then
    printf 'BetaData catalog sync already running: pid=%s\n' "${existing}"
    return 0
  fi
  /usr/bin/nohup /usr/bin/setsid "${PYTHON_BIN}" -u "${SYNC_SCRIPT}" watch \
    "${sync_arguments[@]}" --interval-seconds "${INTERVAL_SECONDS}" \
    </dev/null >>"${LOG_PATH}" 2>&1 &
  local scheduler_pid=$!
  printf '%s\n' "${scheduler_pid}" > "${PID_PATH}"
  sleep 2
  if ! kill -0 "${scheduler_pid}" 2>/dev/null; then
    printf 'BetaData catalog sync failed to start; inspect %s\n' "${LOG_PATH}" >&2
    return 1
  fi
  printf 'BetaData catalog sync started: pid=%s interval=%ss log=%s\n' \
    "${scheduler_pid}" "${INTERVAL_SECONDS}" "${LOG_PATH}"
}

stop_service() {
  local scheduler_pid=""
  if ! scheduler_pid="$(running_pid)"; then
    printf 'BetaData catalog sync is not running\n'
    return 0
  fi
  kill "${scheduler_pid}"
  for _attempt in {1..20}; do
    if ! kill -0 "${scheduler_pid}" 2>/dev/null; then
      rm -f -- "${PID_PATH}"
      printf 'BetaData catalog sync stopped\n'
      return 0
    fi
    sleep 0.25
  done
  printf 'BetaData catalog sync did not stop within 5 seconds: pid=%s\n' "${scheduler_pid}" >&2
  return 1
}

show_status() {
  local scheduler_pid=""
  if scheduler_pid="$(running_pid)"; then
    printf 'running pid=%s interval=%ss\n' "${scheduler_pid}" "${INTERVAL_SECONDS}"
  else
    printf 'stopped\n'
  fi
  if [[ -f "${STATUS_PATH}" ]]; then
    "${PYTHON_BIN}" -m json.tool "${STATUS_PATH}"
  fi
}

case "${ACTION}" in
  start)
    start_service
    ;;
  stop)
    stop_service
    ;;
  restart)
    stop_service
    start_service
    ;;
  run-once)
    "${PYTHON_BIN}" -u "${SYNC_SCRIPT}" once "${sync_arguments[@]}"
    ;;
  status)
    show_status
    ;;
  *)
    printf 'usage: %s {start|stop|restart|run-once|status}\n' "$0" >&2
    exit 2
    ;;
esac
