import json
import os
import time
import traceback
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from enum import Enum

import requests
from dateutil.parser import parse


class ExportTarget(Enum):
    REQUEST_LIST = "request_list"
    TRACE_LIST = "trace_list"
    ALL = "all"


request_list_template: str = """
SELECT 
    `response_duration` AS `Response Delay`,
    auto_instance_0,
    auto_instance_1,
    ip_0,
    ip_1,
    start_time,
    end_time,
    Enum(capture_nic_type),
    Enum(l7_protocol),
    request_type,
    request_domain,
    request_resource,
    Enum(response_status),
    response_code,
    response_exception,
    x_request_id_0,
    x_request_id_1,
    trace_id,
    span_id,
    parent_span_id,
    server_port,
    endpoint,
    toString(_id),
    auto_instance_id_0,
    auto_instance_id_1
FROM 
    `l7_flow_log`
WHERE 
    time >= {start_time}
    AND time <= {end_time}
    AND (`pod_ns_0` = '{namespace}' OR `pod_ns_1` = '{namespace}')
ORDER BY `start_time` ASC
LIMIT {limit}
"""


def _process_row(row: list):
    return [
        f'"{cell}"' if "," in str(cell) or '"' in str(cell) else str(cell)
        for cell in row
    ]


def _transform_request_list_output(text):
    columns = [
        "Response Delay",
        "Client",
        "Server",
        "Client IP",
        "Server IP",
        "Start Time",
        "End Time",
        "Data Type",
        "Protocol",
        "Request Type",
        "Request Domain",
        "Request Resource",
        "Status",
        "Response Code",
        "Response Exception",
        "X-Request-ID",
        "X-Request-ID",
        "Trace ID",
        "Span ID",
        "Parent Span ID",
        "Server Port",
        "Endpoint",
        "Database ID",
        "Client",
        "Server",
    ]
    the_list = []
    for value_list in json.loads(text)["result"].get("values", None) or []:
        new_obj = {}
        for k, v in zip(columns, value_list):
            if k not in new_obj or not len(new_obj[k]):
                new_obj[k] = v
        new_obj["Start Time"] = parse(new_obj["Start Time"])
        new_obj["End Time"] = parse(new_obj["End Time"])
        the_list.append(new_obj)
    return the_list


def export_traces(
    server_base_url: str,
    app_base_url: str,
    start_time_str: str,
    end_time_str: str,
    result_file_name: str,
    export_target_str: str,
):
    export_target = ExportTarget(export_target_str.lower())
    deepflow_server_url = f"{server_base_url}/v1/query"
    deepflow_app_url = (
        f"{app_base_url}/v1/stats/querier/tracing-completion-by-external-app-spans"
    )
    start_time = parse(start_time_str)
    end_time = parse(end_time_str)
    limit = 1000
    namespace = "ray"

    request_obj_list = []

    while start_time.timestamp() < end_time.timestamp():
        data = {
            "db": "flow_log",
            "sql": request_list_template.format(
                limit=limit,
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                namespace=namespace,
            ),
        }
        try:
            # Send the POST request
            server_response = requests.post(deepflow_server_url, data=data)
            if server_response.status_code != 200:
                print("We encountered an error, wait a bit before next request...")
                time.sleep(5)
                continue
            tmp_request_obj_list = _transform_request_list_output(server_response.text)
            if not len(tmp_request_obj_list):
                break
            request_obj_list += tmp_request_obj_list
            start_time = max(
                [obj["Start Time"] for obj in request_obj_list]
            ) + timedelta(microseconds=1)
        except BaseException as e:
            print(
                f"We encountered an error, wait a bit before next request... Error: {e}"
            )
            traceback.print_exc()
            time.sleep(5)
            continue

    print(f"Number of requests gathered from database: {len(request_obj_list)}")

    if export_target == ExportTarget.REQUEST_LIST or export_target == ExportTarget.ALL:
        csv_file_name = f"/app/artifact/{result_file_name}-request-list.csv"
        request_obj_list = sorted(request_obj_list, key=lambda obj: obj["Start Time"])
        with open(csv_file_name, "w") as file:
            # Write header
            file.write(",".join(_process_row(list(request_obj_list[0].keys()))) + "\n")
            # Write rows
            for request_obj in request_obj_list:
                request_obj["Start Time"] = datetime.fromtimestamp(
                    request_obj["Start Time"].timestamp()
                ).isoformat()
                request_obj["End Time"] = datetime.fromtimestamp(
                    request_obj["End Time"].timestamp()
                ).isoformat()
                file.write(",".join(_process_row(list(request_obj.values()))) + "\n")
            print(f"CSV file '{csv_file_name}' written successfully!")

    #####################################################
    if export_target == ExportTarget.TRACE_LIST or export_target == ExportTarget.ALL:
        request_id_resolver = {}
        trace_results = OrderedDict()
        headers = {"Content-Type": "application/json"}
        for request_obj in sorted(
            request_obj_list, key=lambda x: x["Response Delay"], reverse=True
        ):
            if all([len(request_obj[k]) for k in ["X-Request-ID", "Trace ID"]]):
                request_id_resolver[request_obj["Trace ID"]] = max(
                    request_id_resolver.get(request_obj["Trace ID"], ""),
                    request_obj["X-Request-ID"],
                )
            if all([len(request_obj[k]) for k in ["Trace ID", "Span ID"]]):
                obj_trace_id = request_obj["Trace ID"]
                if obj_trace_id not in trace_results:
                    print(obj_trace_id)
                    data = {
                        "max_iteration": 30,
                        "network_delay_us": 3000000,
                        "app_spans": [
                            {
                                "trace_id": obj_trace_id,
                                "span_id": request_obj["Span ID"],
                                "parent_span_id": request_obj["Parent Span ID"],
                                "span_kind": 0,
                                "start_time_us": int(
                                    datetime.fromisoformat(
                                        request_obj["Start Time"]
                                    ).timestamp()
                                    * 1e6
                                ),
                                "end_time_us": int(
                                    datetime.fromisoformat(
                                        request_obj["End Time"]
                                    ).timestamp()
                                    * 1e6
                                ),
                            }
                        ],
                    }
                    while True:
                        # Send the POST request
                        app_response = requests.post(
                            deepflow_app_url, headers=headers, json=data
                        )
                        if app_response.status_code != 200:
                            print(
                                f"We encountered an error while completing data for Trace ID '{obj_trace_id}'!"
                            )
                            time.sleep(5)
                            continue
                        else:
                            trace_results[obj_trace_id] = (
                                app_response.json().get("DATA", {}).get("tracing", [])
                            )
                            break

        target_trace_ids: list[str] = list(
            set(list(request_id_resolver.keys())).intersection(
                set(list(trace_results.keys()))
            )
        )
        print(f"Gathered {len(target_trace_ids)} complete trace(s)")

        new_obj_list = [
            {
                "x_request_id": request_id_resolver[trace_id],
                "tracing": trace_results[trace_id],
            }
            for trace_id in target_trace_ids
        ]
        json_file_name = f"/app/artifact/{result_file_name}-trace-list.json"
        with open(json_file_name, "w") as file:
            json.dump(
                new_obj_list, file, indent=4
            )  # Use indent=4 for pretty formatting
        print(f"JSON file '{json_file_name}' written successfully!")


if __name__ == "__main__":
    export_traces(
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_SERVER_BASE_URL", "http://localhost:20416"
        ),
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_APP_BASE_URL", "http://localhost:20418"
        ),
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_START_TIME_STR",
            (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
                "%Y-%m-%dT%H:%M:%S%z"
            ),
        ),
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_END_TIME_STR",
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        ),
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_RESULT_FILE_NAME", "deepflow-traces"
        ),
        os.environ.get(
            "EXPORT_TRACES_PYTHON_SCRIPT_EXPORT_TARGET", ExportTarget.REQUEST_LIST.value
        ),
    )
