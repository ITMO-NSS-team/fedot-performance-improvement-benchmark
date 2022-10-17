import collections
import json
import operator
import os
import timeit

from copy import deepcopy
from functools import reduce
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from itertools import chain

from fedot.api.main import Fedot
from fedot.core.data.data import DataTypesEnum, InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.opt_history_objects.opt_history import OptHistory

from fedot.core.repository.tasks import Task, TaskTypesEnum
from matplotlib import colors, pyplot as plt

from multiprocessing_benchmark import get_data_from_csv


def _count_pipelines(opt_history: Optional[OptHistory]) -> int:
    if opt_history is not None:
        return reduce(operator.add, map(len, opt_history.individuals), 0)
    return 1


def _get_best_fitness_cv(opt_history: Optional[OptHistory]) -> float:
    if opt_history is not None:
        v = sorted(chain.from_iterable(opt_history.individuals), key=lambda x: x.fitness.value)[0]
        return -v.fitness.value
    return 0


def _show_performance_plot(title: str, x: list, pipelines_count: dict, times: dict, path: str):
    plt.figure()
    plt.title(title)
    plt.xlabel('number of jobs')
    plt.ylabel('correctly evaluated pipelines')

    c_norm = colors.Normalize(vmin=max(min(x) - 0.5, 0), vmax=max(x) + 0.5)
    cm = plt.cm.get_cmap('cool')
    for arg in pipelines_count:
        plt.plot(x, pipelines_count[arg], zorder=1)
        plt.scatter(x, pipelines_count[arg], c=times[arg], cmap=cm, norm=c_norm, zorder=2)

    smp = plt.cm.ScalarMappable(norm=c_norm, cmap=cm)
    smp.set_array([])  # noqa Just to handle 'You must first set_array for mappable' problem
    cb = plt.colorbar(smp)
    cb.ax.set_ylabel('actual time for optimization in minutes', rotation=90)
    plt.xticks(np.arange(min(x), max(x)+1, step=1))
    plt.legend()
    plt.grid()
    plt.savefig(f'{path}/pipelines_count.svg')


def _show_fitness_plot(title: str, x: list, fitnesses: dict, fitnesses_cv: dict, times: dict,
                       path: str):
    plt.figure()
    plt.title(title)
    plt.xlabel('number of jobs')
    plt.ylabel('best fitness')

    c_norm = colors.Normalize(vmin=max(min(x) - 0.5, 0), vmax=max(x) + 0.5)
    cm = plt.cm.get_cmap('cool')
    for arg in fitnesses:
        plt.plot(x, fitnesses[arg],  zorder=1)
        plt.scatter(x, fitnesses[arg], c=times[arg], cmap=cm, norm=c_norm, zorder=2)
    plt.xticks(np.arange(min(x), max(x)+1, step=1))
    smp = plt.cm.ScalarMappable(norm=c_norm, cmap=cm)
    smp.set_array([])  # noqa Just to handle 'You must first set_array for mappable' problem
    cb = plt.colorbar(smp)
    cb.ax.set_ylabel('actual time for optimization in minutes', rotation=90)

    plt.legend()
    plt.grid()
    plt.savefig(f'{path}/fitness.svg')


def _run(processes: List[int], train_data: InputData, test_data: InputData, params: dict, path):
    times = []
    pipelines_count = []
    fitnesses = []
    fitnesses_cv = []
    for n_p in processes:
        for i in range(2):
            if i % 2 == 0:
                # means that the first result in res.json will be with time limit
                params['max_pipeline_fit_time'] = params['timeout']
            else:
                params['max_pipeline_fit_time'] = None
            c_pipelines = 0.
            time = 0.
            fitness = 0.
            fitness_cv = 0.
            mean_range = 3
            for _ in range(mean_range):
                train_data_tmp = deepcopy(train_data)
                test_data_tmp = deepcopy(test_data)
                try:
                    start_time = timeit.default_timer()
                    auto_model = Fedot(problem='classification', logging_level=0, n_jobs=n_p,
                                       **params)
                    auto_model.fit(features=train_data_tmp)
                    auto_model.predict(features=test_data_tmp)
                    time += (timeit.default_timer() - start_time) / 60
                    fitness += auto_model.get_metrics()['roc_auc']
                    c_pipelines += _count_pipelines(auto_model.history)
                    fitness_cv += _get_best_fitness_cv(auto_model.history)
                    auto_model.history.save(f'{path}/{n_p}_{_}_{i}.json')
                except Exception as e:
                    print(e)

            time /= mean_range
            c_pipelines /= mean_range
            fitness /= mean_range
            fitness_cv /= mean_range
            times.append(time)
            pipelines_count.append(c_pipelines)
            fitnesses.append(fitness)
            fitnesses_cv.append(fitness_cv)
    return times, pipelines_count, fitnesses, fitnesses_cv


def compare_n_process(train_data, test_data,  path='res', timeout=60):
    """
    Performs experiment to show how FEDOT works with time limit for single pipeline and without it
    """
    pipelines_count, times, fitnesses, fitnesses_cv = [{timeout: []} for _ in range(4)]
    params = {
        'with_tuning': False, 'num_of_generations': 1000000, 'stopping_after_n_generation': 100000,
        'preset': 'fast_train', 'safe_mode': False, 'cv_folds': 5
    }
    try:
        os.mkdir(path)
    except Exception as e:
        print(e)
    processes = [12]
    params['use_pipelines_cache'] = False
    params['use_preprocessing_cache'] = False
    params['timeout'] = timeout
    _times, _pipelines_count, _fitnesses, _fitnesses_cv = _run(processes, train_data, test_data, params, path=path)
 
    times[timeout] = _times
    pipelines_count[timeout] = _pipelines_count
    fitnesses[timeout] = _fitnesses
    fitnesses_cv[timeout ] = _fitnesses_cv

    all_dict = {'times': times, 'pipelines_count': pipelines_count, 'fitness': fitnesses, 'fitness_cv': fitnesses_cv}

    json.dump(all_dict, open(os.path.join(path, 'res.json'), 'w+'))


if __name__ == "__main__":
    for dataset in os.listdir('data'):
        path_to_dataset = os.path.join('data', dataset)
        problem = 'classification'
        train_data, test_data = get_data_from_csv(data_path=Path(path_to_dataset),
                                                  task_type=TaskTypesEnum.classification)

        compare_n_process(train_data, test_data,  path=f'res_single_pipeline_time_limit/{dataset}', timeout=1)
