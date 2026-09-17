"""Resume a bounded background render of every indexed trajectory video."""
import argparse
import fcntl
import json
import logging
import time
from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path

from vln_gt_video import GTVideoService, readonly

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("workers must be 1..32")
    with args.index.with_suffix(".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with readonly(args.catalog) as db:
            metadata = json.loads(db.execute("SELECT value FROM metadata WHERE key='catalog'").fetchone()[0])
        service = GTVideoService(args.index, metadata, workers=args.workers)
        # Alternate datasets so the first browseable videos cover both R2R and RxR.
        groups = {}
        for identifier, row in service.records.items():
            groups.setdefault(row.get("dataset", "R2R"), []).append(identifier)
        ordered = [ids[i] for i in range(max(map(len, groups.values()))) for ids in groups.values() if i < len(ids)]
        iterator, pending, done = iter(ordered), {}, 0
        started = time.time()
        try:
            while True:
                while len(pending) < args.workers:
                    identifier = next(iterator, None)
                    if identifier is None:
                        break
                    pending[service.pool.submit(service.render, identifier)] = identifier
                if not pending:
                    break
                completed, _ = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for future in completed:
                    identifier = pending.pop(future)
                    try:
                        future.result()
                    except Exception as error:
                        logger.exception("Video generation failed for %s", identifier)
                        print(json.dumps({"record_id": identifier, "error": str(error)[-500:]}), flush=True)
                    done += 1
                if completed:
                    print(json.dumps({"processed": done, "total": len(ordered), "elapsed_seconds": round(time.time()-started), **service.stats()}), flush=True)
        finally:
            service.pool.shutdown(wait=True)
        print(json.dumps({"batch_finished": True, **service.stats()}), flush=True)


if __name__ == "__main__":
    main()
