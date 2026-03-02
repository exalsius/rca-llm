import json
import os
import time
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import parse

TIME_COL: str = "time"


def export_logs(
    base_url: str,
    target_namespace: str,
    start_time_str: str,
    end_time_str: str,
    result_file_name: str,
):
    loki_url = f"{base_url}/loki/api/v1/query_range"
    query = f'{{namespace="{target_namespace}", agent="fluent-bit"}}'
    start_time = parse(start_time_str)
    end_time = parse(end_time_str)
    limit = 1000
    direction = "forward"
    headers = {"Content-Type": "application/json"}

    start_ns = int(start_time.timestamp() * 1e9)
    end_ns = int(end_time.timestamp() * 1e9)
    logs_obj_list = []

    while start_ns < end_ns:
        params = {
            "query": query,
            "start": start_ns,
            "end": end_ns,
            "limit": limit,
            "direction": direction,
            "stream": False,
        }
        print(params)

        try:
            response = requests.get(loki_url, headers=headers, params=params)
            if response.status_code != 200:
                print("We encountered an error, wait a bit before next request...")
                time.sleep(5)
                continue
            data = response.json()

            # Extract logs
            for result_obj in data.get("data", {}).get("result", []):
                pod_name = result_obj.get("stream", {}).get("pod", "N/A")
                container_name = result_obj.get("stream", {}).get("container", "N/A")
                for log_entry in result_obj.get("values", []):
                    log_ts_ns, log_obj_str = log_entry
                    log_obj = eval(log_obj_str)
                    log_file, log_msg = log_obj["true"], log_obj["log"]
                    logs_obj_list.append(
                        {
                            TIME_COL: datetime.fromtimestamp(
                                int(log_ts_ns) / 1e9
                            ).isoformat(),
                            "pod": pod_name,
                            "container": container_name,
                            "file": log_file,
                            "msg": log_msg,
                        }
                    )

            # Update start time to the timestamp of the last log fetched
            if "result" in data.get("data", {}) and data["data"]["result"]:
                last_log_ts_ns = max(
                    [e["values"][-1][0] for e in data["data"]["result"]]
                )
                start_ns = int(last_log_ts_ns) + 1  # Move to the next log
            else:
                break
        except BaseException as e:
            print(
                f"We encountered an error, wait a bit before next request... Error: {e}"
            )
            time.sleep(5)
            continue

    logs_obj_list = sorted(logs_obj_list, key=lambda obj: obj[TIME_COL])
    json_file_name = f"/app/json/{result_file_name}"
    with open(json_file_name, "w") as file:
        json.dump(logs_obj_list, file, indent=4)  # Use indent=4 for pretty formatting
    print(f"JSON file '{json_file_name}' written successfully!")


if __name__ == "__main__":
    export_logs(
        os.environ.get("EXPORT_LOGS_PYTHON_SCRIPT_BASE_URL", "http://localhost:3100"),
        os.environ.get("EXPORT_LOGS_PYTHON_SCRIPT_TARGET_NAMESPACE", "ray"),
        os.environ.get(
            "EXPORT_LOGS_PYTHON_SCRIPT_START_TIME_STR",
            (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        ),
        os.environ.get(
            "EXPORT_LOGS_PYTHON_SCRIPT_END_TIME_STR",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        os.environ.get("EXPORT_LOGS_PYTHON_SCRIPT_RESULT_FILE_NAME", "loki-logs.json"),
    )
