from argparse import Namespace
import traceback
from main import main

if __name__ == "__main__":
    DATASET = "llm-ref-stack-dp"
    DATASET_ROOT = "/app/data_raw"

###################################################
##########    ### METRIC METHODS ##################
###################################################
    metric_methods = []

    metric_methods += [
        "dummy",
        "pc_pagerank",
        "pc_randomwalk",
        "fci_pagerank",
        "fci_randomwalk",
        "granger_pagerank",
        "granger_randomwalk",
        "lingam_pagerank",
        "lingam_randomwalk",
        "causalrca",
        "causalai",
        "microcause",
        "e_diagnosis",
        "baro",
        "rcd",
        "circa",
        "nsigma",
    ]
    metric_methods += [
        "microrca",
        "microscope",
        "monitorrank",
    ]
    
    for method in metric_methods:
        try:
            print(f"[Data-Source=Metric] Running {method}...")
            args = Namespace(
                method=method,
                dataset=DATASET,
                dataset_root=DATASET_ROOT,
                length=60,
                tdelta=0,
                test=False
                )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Metric] Error running {method}: {e}")
            traceback.print_exc()

###################################################
############### TRACE METHODS #####################
###################################################
    trace_methods = [
        "microrank",
        "tracerca"
    ]
    for method in trace_methods:
        try:
            print(f"[Data-Source=Trace] Running {method}...")
            args = Namespace(
                method=method,
                dataset=DATASET,
                dataset_root=DATASET_ROOT,
                length=60,
                tdelta=0,
                test=False
            )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Trace] Error running {method}: {e}")
            traceback.print_exc()

###################################################
############### MULTI-SOURCE METHODS ##############
###################################################
    mm_methods = [
        "mmbaro",
        "pdiagnose"
    ]
    for method in mm_methods:
        try:
            print(f"[Data-Source=Multi-Source] Running {method}...")
            args = Namespace(
                method=method,
                dataset=DATASET,
                dataset_root=DATASET_ROOT,
                length=60,
                tdelta=0,
                test=False
            )
            main(args)
        except Exception as e:
            print(f"[Data-Source=Multi-Source] Error running {method}: {e}")
            traceback.print_exc()