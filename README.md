# Description
This repository is related to paper "Improvement of Computational Performance for Evolutionary AutoML in Heterogeneous
Environment". It used for analysis of improvements in performance while using different performance improvement techniques: caching, parallelization, remote and heterogeneous evaluation.

Tests primarily number of correctly evaluated pipelines for a different training time. Uses caching techniques via relational databases for loading/saving pipelines and data preprocessors.

# Usage
1. Install all dependencies needed for running with `pip install -r requirements.txt`
2. Run with `python benchmark.py`

### Changeable parameters
All the parameters needed to tune the benchmark exists in `benchmark.py` file.
Affects benchmark type:
* `benchmark_number` variable of global scope. Corresponds to the testable function-benchmark.
* Values of `examples_dct` variable of global scope. Matches the parameters in corresponding functions.

Affects benchmark duration:
* `timeouts` variable inside testable functions except `dummy_time_check`. Means `timeout` parameters for training process of FEDOT.
* `mean_range` inside `_run` function. Means averaging the result of FEDOT by running it with the specified number of times.
