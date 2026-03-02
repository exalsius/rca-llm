import logging
import os
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from functools import lru_cache, reduce
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd
import requests
from dateutil.parser import parse

PROM_URL: str = (
    os.environ.get("RCA_COLLECTOR_PROM_URL", "http://localhost:9090")
    + "/api/v1/query_range"
)
K8S_NAMESPACE: str = os.environ.get("RCA_COLLECTOR_K8S_NAMESPACE", "ray")
RESULTS_DIR: Path = Path(
    os.environ.get("RCA_COLLECTOR_RESULT_FOLDER", "/tmp/rca-collector-results")
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIME_COL: str = "time"

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(RESULTS_DIR.joinpath("rca-collector.log"))),
    ],
)
logger = logging.getLogger(__name__)


def _construct_path(suffix):
    return RESULTS_DIR.joinpath(suffix)


@lru_cache()
def _get_node_name_to_node_ip_mapping_dict(
    start_time, end_time, metric_step
) -> Dict[str, str]:
    response = requests.get(
        PROM_URL,
        params={
            "query": "sum(node_uname_info) by (nodename, instance)",
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    metric_results = [
        result["metric"]
        for result in results
        if all(key in result["metric"] for key in ["nodename", "instance"])
    ]
    return {result["nodename"]: result["instance"] for result in metric_results}


@lru_cache()
def _get_pod_name_to_pod_app_label_mapping_dict(
    start_time, end_time, metric_step
) -> Dict[str, str]:
    response = requests.get(
        PROM_URL,
        params={
            "query": f'sum(kube_pod_labels{{namespace="{K8S_NAMESPACE}"}}) by (pod,label_app)',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    metric_results = [
        result["metric"]
        for result in results
        if all(key in result["metric"] for key in ["pod", "label_app"])
    ]
    return {result["pod"]: result["label_app"] for result in metric_results}


@lru_cache()
def _get_pod_address_to_pod_name_mapping_dict(
    start_time, end_time, metric_step
) -> Dict[str, str]:
    response = requests.get(
        PROM_URL,
        params={
            "query": f'sum(kube_pod_info{{namespace="{K8S_NAMESPACE}"}}) by (pod_ip, pod)',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    metric_results = [
        result["metric"]
        for result in results
        if all(key in result["metric"] for key in ["pod_ip", "pod"])
    ]
    return {result["pod_ip"]: result["pod"] for result in metric_results}


def _get_pod_app_label_for_pod_address(
    pod_address: str, pod_name: str, start_time, end_time, metric_step
) -> str:
    pod_ip: str = pod_address.split(":")[0]
    # Pod name is not always available, so we use the pod name from the metric labels by resolving the pod address to pod name
    try:
        pod_name: str = _get_pod_address_to_pod_name_mapping_dict(
            start_time, end_time, metric_step
        )[pod_ip]
    except KeyError:
        # If the pod address is not in the mapping dict, we use the potentially unavailable pod name as a fallback
        pod_name: str = pod_name
    pod_app_label: str = _get_pod_name_to_pod_app_label_mapping_dict(
        start_time, end_time, metric_step
    )[pod_name]
    return pod_app_label


def _prometheus_to_dataframe(results, start_time, end_time, metric_step):
    # Create a list to store all the data
    data = OrderedDict()

    # Process each result
    for result in results:
        # Get the metric labels
        metric_labels = result["metric"]
        target_pod_addresses: List[str] = [
            metric_labels["custom_tag_connection_source_address"],
            metric_labels["custom_tag_connection_destination_address"],
        ]
        target_pod_names: List[str] = [
            metric_labels["custom_tag_inbound_pod_name"],
            metric_labels["custom_tag_outbound_pod_name"],
        ]
        target_pod_app_labels: str = [
            _get_pod_app_label_for_pod_address(
                pod_address, pod_name, start_time, end_time, metric_step
            )
            for pod_address, pod_name in zip(target_pod_addresses, target_pod_names)
        ]
        # Create a column name from the metric labels
        column_name = "_".join(target_pod_app_labels)

        # Process each TIME_COL-value pair
        for ts, value in result["values"]:
            dict_key = int(ts)
            data[dict_key] = data.get(dict_key, OrderedDict())
            data[dict_key][column_name] = float(value) + data[dict_key].get(
                column_name, 0
            )
    # Create DataFrame
    df = pd.DataFrame({TIME_COL: key, **obj} for key, obj in data.items())

    # Pivot the DataFrame to get timestamps as index and metrics as columns
    if not df.empty:
        # Convert TIME_COL to datetime
        df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
        # Convert all numeric columns to float64
        cols_to_cast = df.columns.difference([TIME_COL])
        df[cols_to_cast] = df[cols_to_cast].astype("float64")

    return df


###########################################
############# online analysis #############
###########################################


def latency_source(
    start_time, end_time, metric_step, smoothing_window, percentile: int = 50
):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'histogram_quantile(0.{percentile}, sum(irate(istio_request_duration_milliseconds_bucket{{reporter="waypoint", namespace="{K8S_NAMESPACE}"}}[1m])) by \
                                    (custom_tag_connection_source_address, custom_tag_connection_destination_address, custom_tag_inbound_pod_name, custom_tag_outbound_pod_name, le)) > 0',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    latency_df = _prometheus_to_dataframe(results, start_time, end_time, metric_step)

    if len(latency_df):
        latency_df.set_index(TIME_COL)
        latency_df.to_csv(_construct_path(f"latency_source_{percentile}.csv"))
    return latency_df


def latency_destination(
    start_time, end_time, metric_step, smoothing_window, percentile: int = 50
):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'histogram_quantile(0.{percentile}, sum(irate(istio_request_duration_milliseconds_bucket{{reporter="waypoint", namespace="{K8S_NAMESPACE}"}}[1m])) by \
                                    (custom_tag_connection_source_address, custom_tag_connection_destination_address, custom_tag_inbound_pod_name, custom_tag_outbound_pod_name, le)) > 0',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    latency_df = _prometheus_to_dataframe(results, start_time, end_time, metric_step)

    if len(latency_df):
        latency_df.set_index(TIME_COL)
        latency_df.to_csv(_construct_path(f"latency_destination_{percentile}.csv"))
    return latency_df


def svc_metrics(start_time, end_time, metric_step, smoothing_window):
    response = requests.get(
        PROM_URL,
        params={
            "query": f"sum(rate(container_cpu_usage_seconds_total{{namespace=\"{K8S_NAMESPACE}\", container!~'POD|istio-proxy|fluentbit|'}}[1m])) by (node, pod, instance, container)",
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]

    df_list = []
    for result in results:
        df = pd.DataFrame()
        container_name = result["metric"]["container"]
        pod_name = result["metric"]["pod"]
        node_name = result["metric"]["node"]
        values = result["values"]
        values = list(zip(*values))

        if TIME_COL not in df:
            df[TIME_COL] = values[0]
            df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
        metric = pd.Series(values[1])
        df["ctn_cpu"] = metric
        df["ctn_cpu"] = df["ctn_cpu"].astype("float64")
        # For container network, we use pod name instead of container name
        df["ctn_network"] = ctn_network(start_time, end_time, pod_name, metric_step)
        df["ctn_network"] = df["ctn_network"].astype("float64")
        df["ctn_memory"] = ctn_memory(start_time, end_time, container_name, metric_step)
        df["ctn_memory"] = df["ctn_memory"].astype("float64")
        df["ctn_gpu"] = ctn_gpu(start_time, end_time, container_name, metric_step)
        df["ctn_gpu"] = df["ctn_gpu"].astype("float64")

        node_instance = _get_node_name_to_node_ip_mapping_dict(
            start_time, end_time, metric_step
        )[node_name]

        df_pod_info = pod_info(start_time, end_time, pod_name, metric_step)
        df = pd.merge(df, df_pod_info, how="left", on=TIME_COL)

        df_node_cpu = node_cpu(start_time, end_time, node_instance, metric_step)
        df = pd.merge(df, df_node_cpu, how="left", on=TIME_COL)

        df_node_network = node_network(start_time, end_time, node_instance, metric_step)
        df = pd.merge(df, df_node_network, how="left", on=TIME_COL)

        df_node_memory = node_memory(start_time, end_time, node_instance, metric_step)
        df = pd.merge(df, df_node_memory, how="left", on=TIME_COL)

        df = df.rename(
            columns=lambda col: f"{container_name}_{col}" if col != "time" else col
        )

        df.set_index(TIME_COL)
        df_list.append(df)

    result_df = reduce(
        lambda left, right: pd.merge(left, right, on=TIME_COL, how="outer"), df_list
    )
    result_df = result_df.sort_values(by=TIME_COL).reset_index(drop=True)
    result_df.to_csv(_construct_path("data.csv"))


def ctn_network(start_time, end_time, pod_name, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'((sum(rate(container_network_transmit_packets_total{{namespace="{K8S_NAMESPACE}", pod="{pod_name}"}}[1m])) / 1000 * sum(rate(container_network_transmit_packets_total{{namespace="{K8S_NAMESPACE}", pod="{pod_name}"}}[1m])) / 1000) > 0) or vector(0)',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    values = results[0]["values"]
    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def ctn_memory(start_time, end_time, container_name, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'sum(rate(container_memory_working_set_bytes{{namespace="{K8S_NAMESPACE}", container="{container_name}"}}[1m])) / 1000',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    values = results[0]["values"]
    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def ctn_gpu(start_time, end_time, container_name, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'avg(ray_node_gpus_utilization{{namespace="{K8S_NAMESPACE}", container="{container_name}", ray_io_cluster!~".+"}})',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    if not len(results):
        return pd.Series()
    values = results[0]["values"]
    values = list(zip(*values))
    metric = pd.Series(values[1])
    return metric


def node_network(start_time, end_time, instance, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'rate(node_network_transmit_packets_total{{device="ens6", instance="{instance}"}}[1m]) / 1000',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    values = results[0]["values"]
    values = list(zip(*values))
    df = pd.DataFrame()
    df[TIME_COL] = values[0]
    df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
    df["node_network"] = pd.Series(values[1])
    df["node_network"] = df["node_network"].astype("float64")
    return df


def node_cpu(start_time, end_time, instance, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'sum(rate(node_cpu_seconds_total{{mode != "idle",  mode!= "iowait", mode!~"^(?:guest.*)$", instance="{instance}"}}[1m])) / count(node_cpu_seconds_total{{mode="system", instance="{instance}"}})',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    values = results[0]["values"]
    values = list(zip(*values))
    df = pd.DataFrame()
    df[TIME_COL] = values[0]
    df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
    df["node_cpu"] = pd.Series(values[1])
    df["node_cpu"] = df["node_cpu"].astype("float64")
    return df


def node_memory(start_time, end_time, instance, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'1 - sum(node_memory_MemAvailable_bytes{{instance="{instance}"}}) / sum(node_memory_MemTotal_bytes{{instance="{instance}"}})',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    values = results[0]["values"]
    values = list(zip(*values))
    df = pd.DataFrame()
    df[TIME_COL] = values[0]
    df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
    df["node_memory"] = pd.Series(values[1])
    df["node_memory"] = df["node_memory"].astype("float64")
    return df


def pod_info(start_time, end_time, pod_name, metric_step):
    response = requests.get(
        PROM_URL,
        params={
            "query": f'kube_pod_info{{namespace="{K8S_NAMESPACE}", pod="{pod_name}"}}',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    labels = results[0]["metric"]
    values = results[0]["values"]
    values = list(zip(*values))
    df = pd.DataFrame()
    df[TIME_COL] = values[0]
    df[TIME_COL] = df[TIME_COL].astype("datetime64[s]")
    df["pod_name"] = labels["pod"]
    df["pod_ip"] = labels["pod_ip"]
    df["node_name"] = labels["node"]
    df["node_ip"] = labels["host_ip"]
    return df


# Create Graph
def mpg(start_time, end_time, metric_step):
    DG = nx.DiGraph()
    tuple_set = set()
    # response = requests.get(PROM_URL,
    #                         params={'query': 'sum(istio_tcp_received_bytes_total{source_workload!~\'unknown|kuberay-operator|\', destination_workload!=\'unknown\'}) by (source_app, destination_app)',
    #                                 'start': start_time,
    #                                 'end': end_time,
    #                                 'step': metric_step})
    # results = response.json()['data']['result']

    # for result in results:
    #     metric = result['metric']
    #     source = metric['source_app']
    #     destination = metric['destination_app']
    #     tuple_set.add((source, destination))
    #     DG.add_edge(source, destination)

    #     DG.nodes[source]['type'] = 'service'
    #     DG.nodes[destination]['type'] = 'service'

    response = requests.get(
        PROM_URL,
        params={
            "query": f'sum(rate(istio_requests_total{{reporter="waypoint", namespace="{K8S_NAMESPACE}"}}[1m])) by \
                                    (custom_tag_connection_source_address, custom_tag_connection_destination_address, custom_tag_inbound_pod_name, custom_tag_outbound_pod_name) > 0',
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]

    for result in results:
        metric = result["metric"]
        source = _get_pod_app_label_for_pod_address(
            metric["custom_tag_connection_source_address"],
            metric["custom_tag_inbound_pod_name"],
            start_time,
            end_time,
            metric_step,
        )
        destination = _get_pod_app_label_for_pod_address(
            metric["custom_tag_connection_destination_address"],
            metric["custom_tag_outbound_pod_name"],
            start_time,
            end_time,
            metric_step,
        )
        tuple_set.add((source, destination))
        DG.add_edge(source, destination)

        DG.nodes[source]["type"] = "service"
        DG.nodes[destination]["type"] = "service"

    response = requests.get(
        PROM_URL,
        params={
            "query": f"sum(container_cpu_usage_seconds_total{{namespace=\"{K8S_NAMESPACE}\", container!~'POD|istio-proxy|fluentbit|'}}) by (instance, container)",
            "start": start_time,
            "end": end_time,
            "step": metric_step,
        },
    )
    results = response.json()["data"]["result"]
    for result in results:
        metric = result["metric"]
        if "container" in metric:
            source = metric["container"]
            destination = metric["instance"]
            tuple_set.add((source, destination))
            DG.add_edge(source, destination)

            DG.nodes[source]["type"] = "service"
            DG.nodes[destination]["type"] = "host"

    df = pd.DataFrame(list(tuple_set), columns=["source", "destination"])
    df = df.sort_values(by=["source", "destination"])
    df.to_csv(_construct_path("mpg.csv"))
    return DG


######################################
############# entrypoint #############
######################################


def aggregate_latency(latency_df):
    agg_data = {}
    for node in set(sum([col.split("_") for col in latency_df.columns], [])):
        relevant_cols = [col for col in latency_df.columns if node in col.split("_")]
        agg_data[node] = latency_df[relevant_cols].sum(axis=1)
    aggregated_df = pd.DataFrame(agg_data)
    return aggregated_df


def collect_rca_data():
    start_time_str: str = os.environ.get(
        "RCA_COLLECTOR_START_TIME_STR",
        (datetime.now(timezone.utc) - timedelta(seconds=30)).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        ),
    )
    end_time_str: str = os.environ.get(
        "RCA_COLLECTOR_END_TIME_STR",
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    start_time = parse(start_time_str).timestamp()
    end_time = parse(end_time_str).timestamp()

    metric_step = "5s"
    smoothing_window = 10

    for percentile in [50, 90]:
        latency_df_source = latency_source(
            start_time, end_time, metric_step, smoothing_window, percentile
        )
        latency_df_destination = latency_destination(
            start_time, end_time, metric_step, smoothing_window, percentile
        )
        if len(latency_df_source) or len(latency_df_destination):
            if len(latency_df_source) and len(latency_df_destination):
                latency_df_source.set_index(TIME_COL, inplace=True)
                latency_df_destination.set_index(TIME_COL, inplace=True)
                latency_df_merged = latency_df_destination.add(
                    latency_df_source, fill_value=0
                )
                latency_df_merged.to_csv(
                    _construct_path(f"latency_merged_{percentile}.csv")
                )
                latency_df_aggregated = aggregate_latency(latency_df_merged)
                latency_df_aggregated.to_csv(
                    _construct_path(f"latency_aggregated_{percentile}.csv")
                )

    svc_metrics(start_time, end_time, metric_step, smoothing_window)

    mpg(start_time, end_time, metric_step)


if __name__ == "__main__":
    collect_rca_data()
    logger.info("RCA data collection completed")
