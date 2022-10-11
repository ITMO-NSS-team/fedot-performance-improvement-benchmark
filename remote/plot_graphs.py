import os
import json
import datetime
import numpy as np
from matplotlib import colors, pyplot as plt
from scipy import stats


RESULTS_PATH = "results"
POP_SIZES = (50, 100, 200)
DATETIME_PATTERN = "%Y-%m-%d %H:%M:%S.%f"


def _show_performance_plot(title: str,
                           x: list,
                           pipelines_count: dict,
                           plot_labels: dict,
                           styles: dict):
    plt.figure(figsize=(8, 6), dpi=500)
    plt.title(title)
    plt.xlabel('individuals number')
    plt.ylabel('fit time, sec')

    c_norm = colors.Normalize(vmin=max(min(x) - 0.5, 0), vmax=max(x) + 0.5)
    cm = plt.cm.get_cmap('cool')
    for arg in pipelines_count:
        plt.plot(
            x,
            pipelines_count[arg],
            label=plot_labels[arg],
            zorder=1,
            linestyle=styles[arg]
        )
        plt.scatter(x, pipelines_count[arg], cmap=cm, norm=c_norm, zorder=2)

    smp = plt.cm.ScalarMappable(norm=c_norm, cmap=cm)
    smp.set_array([])

    plt.legend()
    plt.grid()
    plt.show()


EDGE_COLORS = {
    "pod": "purple",
    "container": "blue",
    "computing": "darkgreen"
}
FILL_COLORS = {
    "pod": "orchid",
    "container": "cyan",
    "computing": "lightgreen"
}


def _show_overheads_plot(data: dict):
    plt.figure(figsize=(8, 4), dpi=500)
    plt.title("Overheads and computing time")
    plt.xlabel('Elapsed time, sec')
    data_keys = sorted(data.keys())

    plots = [
        data[pop_size][kind]
        for kind in ["pod", "container", "computing"]
        for pop_size in data_keys
    ]

    labels = [
        label.format(pop_size=pop_size)
        for label in [
            "Pod starting time ({pop_size})",
            "Container starting time ({pop_size})",
            "Computing time ({pop_size})"
        ]
        for pop_size in data_keys
    ]

    edge_colors = [
        EDGE_COLORS[kind]
        for kind in ["pod", "container", "computing"]
        for pop_size in data_keys
    ]

    fill_colors = [
        FILL_COLORS[kind]
        for kind in ["pod", "container", "computing"]
        for pop_size in data_keys
    ]

    box = plt.boxplot(
        plots,
        vert=False,
        labels=labels,
        flierprops=dict(markersize=5),
        medianprops=dict(linewidth=2, color="firebrick"),
        patch_artist=True
    )

    for patch, fill_color, edge_color in zip(
            box['boxes'], fill_colors, edge_colors
    ):
        patch.set_facecolor(fill_color)
        patch.set_edgecolor(edge_color)

    for element in ['whiskers', 'caps']:
        for i, line in enumerate(box[element]):
            line.set_color(edge_colors[i // 2])

    for line, edge_color, fill_color in zip(box['fliers'], edge_colors, fill_colors):
        line._markeredgecolor = edge_color
        line._markerfacecolor = fill_color

    plt.grid(axis='x', which="major")
    plt.grid(axis='x', which="minor", linestyle='--')
    plt.minorticks_on()

    plt.show()


def _show_timeline(data: dict):
    data_keys = sorted(data.keys())
    for pop_size, exp in data.items():
        plt.figure(figsize=(8, 2), dpi=500)
        plt.title(f"Fit timeline (Population size = {pop_size})")
        plt.xlabel('Elapsed time, sec')

        plt.barh(
            ["Fetching results", "Computing", "Requests"],
            [exp["fetching_end"], exp["computing_end"], exp["requests_end"]],
            left=[exp["requests_end"], exp["requests_start"], exp["requests_start"]]
        )

        plt.grid(axis='x', which="major")
        plt.grid(axis='x', which="minor", linestyle='--')
        plt.minorticks_on()

        plt.show()


if __name__ == "__main__":

    for requested_cpu in [0.1, 1.0]:
        data = dict()
        for file in list(
                filter(
                    lambda file: file.endswith(".json") and file.startswith(f"cpu_{requested_cpu}"),
                    os.listdir(RESULTS_PATH)
                )
        ):
            with open(f"{RESULTS_PATH}/{file}", "r") as f:
                data[file] = json.loads(f.read())

        # Average computing time without results downloading
        avg_pop = {pop_size: list() for pop_size in POP_SIZES}
        for exp in data.values():
            if exp["pop_size"] in POP_SIZES:
                avg_pop[exp["pop_size"]].append(
                    (
                            datetime.datetime.strptime(exp["max_container_finish_time"].replace("+00:00", ".000000"), DATETIME_PATTERN) -
                            datetime.datetime.strptime(exp["client_train_start_time"], DATETIME_PATTERN)
                    ).total_seconds()
                )

        _show_performance_plot(
            title="Remote fit performance depends on population size",
            x=POP_SIZES,
            pipelines_count={
                "actual": [
                    np.average(avg_pop[key]) for key in sorted(avg_pop.keys())
                ],
                "linear": [
                    np.average(avg_pop[50]) * c for c in [1, 2, 4]
                ]
            },
            plot_labels={"actual": "remote fit time", "linear": "linear fit time"},
            styles={"actual": "solid", "linear": "dashed"}
        )

        # Overheads and computing time boxplots. Take only 1st experiment in series.
        overheads = dict()
        for exp in data.values():
            pop_size = exp["pop_size"]
            if pop_size not in overheads and pop_size in POP_SIZES:
                overheads[pop_size] = {
                    "pod": exp["pods_starting_overheads"],
                    "container": exp["container_starting_overheads"],
                    "computing": exp["computing_times"]
                }

        _show_overheads_plot(overheads)

        timeline = dict()
        for exp in data.values():
            pop_size = exp["pop_size"]
            if pop_size not in timeline and pop_size in POP_SIZES:
                start_point = datetime.datetime.strptime(exp["client_train_start_time"], DATETIME_PATTERN)
                timeline[pop_size] = {
                    "requests_start": 0,
                    "requests_end": (
                            datetime.datetime.strptime(exp["requests_finished_time"], DATETIME_PATTERN) - start_point
                    ).total_seconds(),
                    "computing_end": (
                            datetime.datetime.strptime(exp["max_container_finish_time"].replace("+00:00", ".000000"), DATETIME_PATTERN) - start_point
                    ).total_seconds(),
                    "fetching_end": (
                            datetime.datetime.strptime(exp["client_train_end_time"], DATETIME_PATTERN) - start_point
                    ).total_seconds()
                }

        _show_timeline(timeline)
