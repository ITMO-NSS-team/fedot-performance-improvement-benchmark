import os
import logging
import numpy as np
import json
import nest_asyncio
from datetime import datetime
import operator
from copy import deepcopy
from functools import reduce
from typing import List, Optional
from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.pipelines.node import PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.utils import fedot_project_root
from fedot.remote.infrastructure.clients.test_client import TestClient
from fedot.remote.infrastructure.clients.datamall_client import DataMallClient
from fedot.core.optimisers.opt_history import OptHistory
from fedot.remote.remote_evaluator import RemoteEvaluator, RemoteTaskParams
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from kubernetes import client, config


nest_asyncio.apply()

config.load_kube_config(context="default")
v1 = client.CoreV1Api()

MAX_PARALLEL = 800
PRESET = "fast_train"

CONNECT_PARAMS = {
    'FEDOT_LOGIN': 'test1234',
    'FEDOT_PASSWORD': '*******',
    'AUTH_SERVER': 'http://10.32.0.51:30880/b',
    'CONTR_SERVER': 'http://10.32.0.51:30880/models-controller',
    'PROJECT_ID': '82',
    'DATA_ID': '131'
}

DEFAULT_EXEC_PARAMS = {
    'container_input_path': "/home/FEDOT/input_data_dir",
    'container_output_path': "/home/FEDOT/output_data_dir",
    'container_config_path': "/home/FEDOT/.config",
    'container_image': "fedot:dm-10-20220927",
    'timeout': 300,
    "nodes_blacklist": ["node18.bdcl", "node23.bdcl", "node24.bdcl", "node10.bdcl"],
    "keep_results": False,
    "is_fs_mounted": False
}

REMOTE_TASK_PARAMS = RemoteTaskParams(
    mode='remote',
    dataset_name='scoring_train',
    task_type='Task(TaskTypesEnum.classification)',
    max_parallel=MAX_PARALLEL
)

TRAIN_DATA = InputData.from_csv("data/scoring_train.csv",
                                task=Task(TaskTypesEnum.classification))
TEST_DATA = InputData.from_csv("data/scoring_test.csv",
                               task=Task(TaskTypesEnum.classification))


def run(timeout, pop_size, num_generations, exec_params):
    dm_client = DataMallClient(
        connect_params=CONNECT_PARAMS,
        exec_params=exec_params,
        output_path="./output"
    )

    evaluator = RemoteEvaluator()
    evaluator.init(
        client=dm_client,
        remote_task_params=REMOTE_TASK_PARAMS
    )

    composer_params = {
        'cv_folds': None,
        'with_tuning': False,
        'pop_size': pop_size,
        'timeout': timeout,
        'num_of_generations': num_generations,
        'genetic_scheme': 'steady_state'
    }

    fedot_input = {
        'problem': "classification",
        'seed': 0,
        'preset': PRESET,
        **composer_params,
        'logging_level': 20,
    }

    auto_model = Fedot(**fedot_input)

    start = datetime.utcnow()
    model = auto_model.fit(features=TRAIN_DATA, target='target')
    end = datetime.utcnow()

    auto_model.predict_proba(features=TEST_DATA)
    metrics = auto_model.get_metrics()

    d = {
        "metrics": metrics,
        "group_id": dm_client.group_id,
        "create_task_overheads": [ts.total_seconds() for ts in dm_client.create_task_overheads],
        "download_result_overheads": [ts.total_seconds() for ts in dm_client.download_result_overheads],
        "start_train_time": str(start),
        "client_train_start_time": str(dm_client.train_start_time),
        "requests_finished_time": str(dm_client.requests_finished_time),
        "client_train_end_time": str(dm_client.train_end_time),
        "end_train_time": str(end),
        "pop_size": pop_size,
        "num_generations": num_generations,
        "timeout": timeout
    }

    return d


if __name__ == "__main__":
    results = list()

    # CPU limits have been configured on the cluster side based on requested cpu.
    # 0.1 => CPU limits = 0.2, 1.0 => CPU is unlimited
    for requested_cpu in [0.1, 1.0]:

        exec_params = deepcopy(DEFAULT_EXEC_PARAMS)
        exec_params["cpu"] = requested_cpu

        for pop_size in [50, 100, 200]:
            for i in range(4):
                print(f"=================== Attempt #{i} ===================")
                print(f"{pop_size=}")
                result = run(timeout=20, pop_size=pop_size, num_generations=1, exec_params=exec_params)

                # Extract pods events timestamps
                pods = v1.list_namespaced_pod(
                    namespace="fedot-test",
                    label_selector=f"models-operator-execution-group-id={result['group_id']}"
                )

                pods_starting_overheads = []
                container_starting_overheads = []
                computing_times = []
                total_times = []
                min_pod_creation_time = min([pod.metadata.creation_timestamp for pod in pods.items])
                max_container_finish_time = max(
                    [pod.status.container_statuses[0].state.terminated.finished_at for pod in pods.items]
                )

                for pod in pods.items:
                    pod_creation_ts = pod.metadata.creation_timestamp
                    pod_start_ts = pod.status.start_time
                    container_start_ts = pod.status.container_statuses[0].state.terminated.started_at
                    container_finish_ts = pod.status.container_statuses[0].state.terminated.finished_at

                    pod_starting_overhead: "datetime.timedelta" = (pod_start_ts - pod_creation_ts).total_seconds()
                    container_starting_overhead = (container_start_ts - pod_start_ts).total_seconds()
                    computing_time = (container_finish_ts - container_start_ts).total_seconds()
                    total_time = (container_finish_ts - pod_creation_ts).total_seconds()

                    pods_starting_overheads.append(pod_starting_overhead)
                    container_starting_overheads.append(container_starting_overhead)
                    computing_times.append(computing_time)
                    total_times.append(total_time)


                result["pods_starting_overheads"] = pods_starting_overheads
                result["container_starting_overheads"] = container_starting_overheads
                result["computing_times"] = computing_times
                result["total_times"] = total_times
                result["min_pod_creation_time"] = str(min_pod_creation_time)
                result["max_container_finish_time"] = str(max_container_finish_time)

                results.append(result)

                with open(
                        f"results/cpu_{requested_cpu}_{pop_size}_{i}_{int(datetime.utcnow().timestamp())}.json",
                        "w"
                ) as f:
                    f.write(json.dumps(result))

                _ = v1.delete_collection_namespaced_pod(
                    namespace="fedot-test",
                    label_selector=f"models-operator-execution-group-id={result['group_id']}"
                )
