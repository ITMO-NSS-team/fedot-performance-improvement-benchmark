import collections
import logging
import operator
import timeit

from collections import defaultdict
from copy import deepcopy
from functools import reduce
from statistics import mean
from timeit import repeat
from typing import List, Optional

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.optimisers.opt_history import OptHistory
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams
from matplotlib import colors, pyplot as plt

from core import get_dataset


def _count_pipelines(opt_history: Optional[OptHistory]) -> int:
    if opt_history is not None:
        return reduce(operator.add, map(len, opt_history.individuals), 0)
    return 0


def _show_performance_plot(title: str, x: list, pipelines_count: dict, times: dict, plot_labels: dict):
    plt.title(title)
    plt.xlabel('timeout in minutes')
    plt.ylabel('correctly evaluated pipelines')

    c_norm = colors.Normalize(vmin=max(min(x) - 0.5, 0), vmax=max(x) + 0.5)
    cm = plt.cm.get_cmap('cool')
    for arg in pipelines_count:
        plt.plot(x, pipelines_count[arg], label=plot_labels[arg], zorder=1)
        plt.scatter(x, pipelines_count[arg], c=times[arg], cmap=cm, norm=c_norm, zorder=2)

    smp = plt.cm.ScalarMappable(norm=c_norm, cmap=cm)
    smp.set_array([])  # noqa Just to handle 'You must first set_array for mappable' problem
    cb = plt.colorbar(smp)
    cb.ax.set_ylabel('actual time for optimization in minutes', rotation=90)

    plt.legend()
    plt.grid()
    plt.show()


def dummy_time_check():
    composer_params = {
        'with_tuning': False,
        'validation_blocks': 1,
        'cv_folds': None,

        'max_depth': 4, 'max_arity': 2, 'pop_size': 3,
        'timeout': None, 'num_of_generations': 5
    }

    for use_cache in [False, True]:
        print(f'Using cache mode: {use_cache}')
        for task_type in ['ts_forecasting', 'regression', 'classification']:
            preset = 'best_quality'
            fedot_input = {'problem': task_type, 'seed': 42, 'preset': preset, 'verbose_level': logging.NOTSET,
                           'timeout': composer_params['timeout'],
                           'use_pipelines_cache': use_cache, 'use_preprocessing_cache': use_cache,
                           **composer_params}
            if task_type == 'ts_forecasting':
                fedot_input['task_params'] = TsForecastingParams(forecast_length=30)
            train_data, test_data, _ = get_dataset(task_type)

            def check():
                Fedot(**fedot_input).fit(features=train_data, target='target')

            print(f"task_type={task_type}, mean_time={mean(repeat(check, repeat=15, number=1))}")


def _run(timeouts: List[int], train_data: InputData, test_data: InputData, params: dict):
    times = []
    pipelines_count = []
    for timeout in timeouts:
        c_pipelines = 0.
        time = 0.
        mean_range = 3
        cache_effectiveness = collections.Counter()
        fitness = collections.Counter()
        for _ in range(mean_range):
            train_data_tmp = deepcopy(train_data)
            start_time = timeit.default_timer()
            auto_model = Fedot(**params, timeout=timeout)
            auto_model.fit(features=train_data_tmp)
            time += (timeit.default_timer() - start_time) / 60
            auto_model.predict_proba(features=test_data_path)
            fitness += auto_model.get_metrics()
            c_pipelines += _count_pipelines(auto_model.history)
            if params.get('use_pipelines_cache') and auto_model.api_composer.pipelines_cache.effectiveness_ratio:
                cache_effectiveness += auto_model.api_composer.pipelines_cache.effectiveness_ratio
            elif (params.get('use_preprocessing_cache')
                  and auto_model.api_composer.preprocessing_cache.effectiveness_ratio):
                cache_effectiveness += auto_model.api_composer.preprocessing_cache.effectiveness_ratio

        time /= mean_range
        c_pipelines /= mean_range
        times.append(time)
        pipelines_count.append(c_pipelines)
        cache_effectiveness = {k: v / mean_range for k, v in cache_effectiveness.items()}
        fitness = {k: v / mean_range for k, v in fitness.items()}

        print((
            f'\tTimeout: {timeout}'
            f', number of pipelines: {c_pipelines}, elapsed time: {time:.3f}'
            f', fitness: {fitness}'
        ))
    return times, pipelines_count


def test_cache(problem: str, train_data: InputData, test_data: InputData,
                    n_jobs: int = 1, test_both: bool = False, test_pipelines: bool = True):
    """
    Performs experiment to show how caching helps in fitting FEDOT model
    """
    pipelines_count, times = [{False: [], True: []} for _ in range(2)]
    plot_labels = {False: 'without cache', True: 'with cache'}
    preset = 'fast_train'
    fedot_params = {
        'problem': problem, 'preset': preset, 'with_tuning': False,
        'logging_level': logging.CRITICAL, 'show_progress': False,
        'n_jobs': n_jobs, 'seed': 42
    }
    timeouts = [1, 2, 3, 4]
    for use_cache in [True, False]:
        print(f'Using cache: {use_cache}')
        if test_both:
            fedot_params['use_pipelines_cache'] = fedot_params['use_preprocessing_cache'] = use_cache
        elif test_pipelines:
            fedot_params['use_pipelines_cache'] = use_cache
            fedot_params['use_preprocessing_cache'] = False
        else:
            fedot_params['use_pipelines_cache'] = False
            fedot_params['use_preprocessing_cache'] = use_cache
        _times, _pipelines_count = _run(timeouts, train_data, test_data, fedot_params)
        times[use_cache] = _times
        pipelines_count[use_cache] = _pipelines_count
    title = f'Cache performance with n_jobs={n_jobs}'
    if not test_both:
        if test_pipelines:
            title = 'Pipelines ' + title[0].lower() + title[1:]
        else:
            title = 'Preprocessing ' + title[0].lower() + title[1:]
    _show_performance_plot(title, timeouts, pipelines_count, times, plot_labels)


def test_log(problem: str, train_data: InputData, test_data: InputData, n_jobs: int = 1):
    """
    Performs experiment to show how logger mp synchronization influences FEDOT performance
    """
    pipelines_count, times = [{False: [], True: []} for _ in range(2)]
    plot_labels = {False: 'mp logger disabled', True: 'mp logger enabled'}
    preset = 'fast_train'
    fedot_params = {
        'problem': problem, 'preset': preset, 'with_tuning': False,
        'logging_level': logging.CRITICAL, 'show_progress': False,
        'n_jobs': n_jobs, 'seed': 42
    }
    timeouts = [2, 3, 4]
    for use_sync in [False, True]:
        print(f'Using synchronization: {use_sync}')
        fedot_params['sync_logs_in_mp']
        _times, _pipelines_count = _run(timeouts, train_data, test_data, fedot_params)
        times[use_sync] = _times
        pipelines_count[use_sync] = _pipelines_count
    _show_performance_plot(f'Log performance with n_jobs={n_jobs}', timeouts, pipelines_count, times, plot_labels)


def compare_cache_sp_vs_mp(problem: str, train_data, test_data, n_jobs: int = -1,
                           test_both: bool = False, test_pipelines: bool = True):
    """
    Performs experiment to show how one-process FEDOT cacher compares to the multiprocessed
    """
    assert n_jobs != 1, 'This test uses multiprocessing, so you should have > 1 processors'
    pipelines_count, times = [{1: [], n_jobs: []} for _ in range(2)]
    plot_labels = {1: 'one process', n_jobs: f'{n_jobs} processes'}
    fedot_params = {
        'problem': problem, 'preset': 'fast_train', 'with_tuning': False,
        'logging_level': logging.CRITICAL, 'show_progress': False,
        'seed': 42
    }
    timeouts = [1, 2, 3, 4, 5]
    for _n_jobs in [1, n_jobs]:
        print(f'Processes used: {_n_jobs}')
        if test_both:
            fedot_params['use_pipelines_cache'] = fedot_params['use_preprocessing_cache'] = True
        elif test_pipelines:
            fedot_params['use_pipelines_cache'] = True
            fedot_params['use_preprocessing_cache'] = False
        else:
            fedot_params['use_pipelines_cache'] = False
            fedot_params['use_preprocessing_cache'] = True
        fedot_params['n_jobs'] = _n_jobs
        _times, _pipelines_count = _run(timeouts, train_data, test_data, fedot_params)
        times[_n_jobs] = _times
        pipelines_count[_n_jobs] = _pipelines_count
    title = f'Cache performance comparison between one process and {n_jobs}'
    if not test_both:
        if test_pipelines:
            title = 'Pipelines ' + title[0].lower() + title[1:]
        else:
            title = 'Preprocessing ' + title[0].lower() + title[1:]
    _show_performance_plot(title, timeouts, pipelines_count, times, plot_labels)


if __name__ == "__main__":
    problem = 'classification'
    if problem == 'classification':
        train_data_path = 'core/data/scoring/scoring_train.csv'
        test_data_path = 'core/data/scoring/scoring_test.csv'
        train_data = InputData.from_csv(train_data_path,
                                        task=Task(TaskTypesEnum.classification)).subset_range(0, 4000)
        test_data = InputData.from_csv(test_data_path,
                                       task=Task(TaskTypesEnum.classification)).subset_range(0, 4000)
    elif problem == 'regression':
        data_path = 'core/data/cholesterol/cholesterol.csv'
        data = InputData.from_csv(data_path,
                                  task=Task(TaskTypesEnum.regression))
        train_data, test_data = train_test_data_setup(data)

    examples_dct = defaultdict(lambda: (lambda: print('Wrong example number option'),))
    examples_dct.update({
        1: (dummy_time_check,),
        2: (test_cache, problem, train_data, test_data, 1, False),
        3: (compare_cache_sp_vs_mp, problem, train_data, test_data, -1, False),
        4: (test_log, problem, train_data, test_data, -1)
    })
    benchmark_number = 2
    func, *args = examples_dct[benchmark_number]

    func(*args)
