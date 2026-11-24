"""Container-side self-deletion; complements monitoring, not a billing guarantee.

Run as a separate process before any setup or model download. Credentials are
read from the environment and never included in logs or checkpoint artifacts.
Network/provider failure can still delay deletion; keep an external monitor.
"""
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def delete_pod(pod_id, key):
    request = Request(
        f"https://api.runpod.io/v2/pods/{pod_id}", method="DELETE",
        headers={"Authorization": f"Bearer {key}"},
    )
    try:
        with urlopen(request, timeout=15) as response:
            return 200 <= response.status < 300
    except HTTPError as exc:
        return exc.code == 404
    except (URLError, TimeoutError, OSError):
        return False


def main():
    deadline = float(os.environ["GUARD_DEADLINE_UNIX"])
    pod_id = os.environ["RUNPOD_POD_ID"]
    key = os.environ["GUARD_API_KEY"]
    status = Path("/workspace/cost-guard.json")
    status.write_text(json.dumps({"armed": True, "pid": os.getpid(),
                                  "pod_id": pod_id, "deadline": deadline}))
    while time.time() < deadline:
        time.sleep(min(5, max(0, deadline - time.time())))
    while not delete_pod(pod_id, key):
        time.sleep(5)


if __name__ == "__main__":
    main()
