import os
import time
from datetime import datetime, timedelta, timezone

import requests
from dateutil.parser import parse


# filter metrics from the config file
def get_metrics_name():
    # filter names based on input
    metrics_file = "metrics.txt"

    with open("/app/config/" + metrics_file) as input_metrics:
        lines = input_metrics.read().splitlines()
    new_names = list(set(lines))
    return new_names


def export_metrics(
    base_url: str, start_time_str: str, end_time_str: str, result_file_name: str
):
    prometheus_url = f"{base_url}/api/v1/query_range"
    start_time = parse(start_time_str)
    end_time = parse(end_time_str)

    metric_names = get_metrics_name()
    results = {}

    for raw_metric_name in sorted(metric_names):
        metric_name_parts = raw_metric_name.split(" @@@ ")
        metric_name, metric_query = metric_name_parts[0], metric_name_parts[-1]
        print(f"Attempting to export metric: '{metric_name}'")
        while True:
            try:
                response = requests.get(
                    prometheus_url,
                    params={
                        "query": metric_query,
                        "start": start_time.timestamp(),
                        "end": end_time.timestamp(),
                        "step": "5s",
                    },
                    verify=False,
                )
                if response.status_code != 200:
                    print("We encountered an error, wait a bit before next request...")
                    time.sleep(5)
                    continue
                for sub_result in response.json()["data"]["result"]:
                    title = metric_name
                    try:
                        title = sub_result["metric"]["__name__"]
                    except KeyError:
                        pass
                    list_of_tuples = sub_result["values"]
                    results[title] = results.get(title, []) + list_of_tuples
                break
            except BaseException as e:
                print(
                    f"We encountered an error, wait a bit before next request... Error: {e}"
                )
                time.sleep(5)
                continue

    # Consolidate data
    timestamps = sorted(set(ts for metric in results.values() for ts, _ in metric))
    data_by_timestamp = {ts: {} for ts in timestamps}

    for metric_name, values in results.items():
        for timestamp, value in values:
            data_by_timestamp[timestamp][metric_name] = value

    csv_file_name = f"/app/csv/{result_file_name}"
    # Write to CSV file
    with open(csv_file_name, "w") as file:
        # Write header
        header = ["timestamp"] + list(results.keys())
        file.write(",".join(header) + "\n")

        # Write rows
        for timestamp in timestamps:
            timestamp_new = datetime.fromtimestamp(timestamp).isoformat()
            row = [str(timestamp_new)] + [
                str(data_by_timestamp[timestamp].get(metric, ""))
                for metric in results.keys()
            ]
            file.write(",".join(row) + "\n")
    print(f"CSV file '{csv_file_name}' written successfully!")


if __name__ == "__main__":
    export_metrics(
        os.environ.get(
            "EXPORT_METRICS_PYTHON_SCRIPT_BASE_URL", "http://localhost:9090"
        ),
        os.environ.get(
            "EXPORT_METRICS_PYTHON_SCRIPT_START_TIME_STR",
            (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        ),
        os.environ.get(
            "EXPORT_METRICS_PYTHON_SCRIPT_END_TIME_STR",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        os.environ.get(
            "EXPORT_METRICS_PYTHON_SCRIPT_RESULT_FILE_NAME", "prometheus-metrics.csv"
        ),
    )
