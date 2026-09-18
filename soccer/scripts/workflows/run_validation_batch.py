"""Run frozen new clips through the durable workflow, then score locally."""

import json
import time
from pathlib import Path

from soccerviz.vision.video_jobs import inspect, submit


def main():
    root = Path("artifacts/workflow-corpus")
    path = root / "jobs.json"
    jobs = json.loads(path.read_text()) if path.exists() else {}
    for name in ["SNGS-022", "SNGS-050", "SNGS-079", "SNGS-093"]:
        if name not in jobs:
            job, _ = submit(root / "videos5hz" / name / "source.mp4", Path("artifacts"),
                            seconds=6, jersey=True)
            jobs[name] = {"job_id": job}
            path.write_text(json.dumps(jobs, indent=2) + "\n")
            print(name, job, flush=True)
        for _ in range(240):
            state = inspect(jobs[name]["job_id"], Path("artifacts"))
            if state["status"] in {"completed", "failed"}:
                jobs[name].update({k: state[k] for k in ["status", "video_run"] if k in state})
                path.write_text(json.dumps(jobs, indent=2) + "\n")
                if state["status"] == "failed":
                    raise RuntimeError(json.dumps(state))
                print(name, state["status"], state["video_run"], flush=True)
                break
            time.sleep(3)
        else:
            raise TimeoutError(f"Job still running; resume with {jobs[name]['job_id']}")


if __name__ == "__main__":
    main()
