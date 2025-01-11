from enum import Enum
from math import pi
from typing import Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from consts import dataset_name_map, model_map, model_to_color, scandeval_categories


def load_agg_metrics(
    run_id: str = "final",
    run_folder: str = "RUNS",
    ext: str = "xlsx",
    group: bool = False,
    filter_prompt="full_description",
) -> pd.DataFrame:
    # filepath = f"{run_folder}/agg_metrics_{run_id}.{ext}"
    # ignore index in total
    if ext == "csv":
        df = pd.read_csv(
            f"{run_folder}/agg_metrics_{run_id}.{ext}", sep=";", index_col=0
        )
        # df = df.reset_index()
    else:
        df = pd.read_excel(f"{run_folder}/agg_metrics_{run_id}.{ext}", index_col=0)
    if filter_prompt:
        df = df[df["prompts"] == filter_prompt]
    if group:
        df = group_metrics(df)
    return df


# Group by prompts and calculate the mean, min, max, and std of accuracy
def group_metrics(agg_metrics: pd.DataFrame) -> pd.DataFrame:
    agg_df = agg_metrics.groupby(
        [
            "temperature",
            "test_mode",
            "refine",
            "prompts",
            "models",
            "max_task_tokens",
        ],
        observed=False,
    )[
        [
            "accuracy",
            # "avg_duration_sec",
            # "avg_vram_used",
        ]
    ].agg(
        ["mean", "min", "max", "std"]
    )
    agg_df.columns = ["_".join(col).strip() for col in agg_df.columns.values]
    return agg_df


def get_best_run(agg_metrics: pd.DataFrame):
    max_accuracy_index = agg_metrics["accuracy"].idxmax()
    best_run = agg_metrics.loc[max_accuracy_index]["run"]
    return agg_metrics.loc[max_accuracy_index].to_frame()


def get_models(agg_df: pd.DataFrame):
    models = agg_df.index.get_level_values("models").unique().tolist()
    return models


def load_scandeval(date: str):
    sdval = pd.read_csv(f"norwegian-nlg-{date}.csv")
    # sdval = sdval.rename(columns={"model_id": "model"})
    # process_fn = lambda x: x.split(" (")[0]
    # sdval["model"] = sdval["model"].apply(process_fn)
    # sdval = sdval[sdval["model"].isin(models)]
    sdval.columns = sdval.columns.str.replace("-no", "")
    # sdval.columns = sdval.columns.str.replace("-nb", "")

    return sdval


def brage_statistics(
    df: pd.DataFrame, models: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.reset_index()
    df = df[df["models"].isin(models)]
    df = df[
        [
            "prompts",
            "models",
            "accuracy_mean",
            "accuracy_min",
            "accuracy_max",
            "accuracy_std",
        ]
    ]
    brage_full_desc = df[df["prompts"] == "full_description"]
    brage_keywords = df[df["prompts"] == "instruction_w_keywords"]
    return brage_full_desc, brage_keywords


brage_cols = "brage_full_desc brage_keywords".split()


def get_brage_accuracy(scandeval_df, model):
    if model not in scandeval_df.index:
        return None
    accuracies = scandeval_df.loc[model][brage_cols].values
    return np.average(accuracies).round(2).item()


from collections import OrderedDict

linestyles = OrderedDict(
    [
        # ("solid", (0, ())),
        ("dotted", (0, (1, 5))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("dashed", (0, (5, 5))),
        ("densely dotted", (0, (1, 1))),
        ("densely dashed", (0, (5, 1))),
        ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("densely dashdotted", (0, (3, 1, 1, 1))),
        ("loosely dashed", (0, (5, 10))),
        ("loosely dotted", (0, (1, 10))),
        ("loosely dashdotted", (0, (3, 10, 1, 10))),
        ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
        ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
    ]
)


def spider_plot(
    df: pd.DataFrame,
    models: list,
    which_metrics: List[str],
    linestyles: List[str] = None,
    title: str = "Benchmark results",
):
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    if not linestyles:
        linestyles = ["solid"] * len(models)

    for idx, model in enumerate(models):
        metrics = df.loc[model][which_metrics].to_dict()
        categories = list(metrics.keys())
        categories = [dataset_name_map[c] for c in categories]
        N = len(categories)
        values = list(metrics.values())
        values = [max(0, v) for v in values]
        values += values[:1]

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle=linestyles[idx],
            label=model_map[model],
            color=model_to_color[model],
        )
        ax.fill(angles, values, alpha=0.1, color=model_to_color[model])
        ax.set_ylim(0, 100)
        ax.set_rgrids([20, 40, 60, 80, 100], angle=30, fontsize=16)

    plt.xticks(angles[:-1], categories, color="black", size=18, fontweight="bold")
    # plt.legend(
    #     loc="upper right",
    #     bbox_to_anchor=(0.5, 0.5),
    #     fontsize=12,
    #     frameon=True,
    #     borderaxespad=0,
    # )

    legend_colors = []
    for model in models:
        legend_colors.append((model_map[model], model_to_color[model]))

    legend_patches = []
    for patch_idx, (label, color) in enumerate(legend_colors):
        legend_patches.append(
            mpatches.Patch(
                facecolor="None",
                edgecolor=color,
                linewidth=2,
                label=label,
                linestyle=linestyles[patch_idx],
            )
        )

    plt.tight_layout()
    plt.legend(
        handles=legend_patches,
        loc="lower left",
        ncol=2,
        bbox_to_anchor=(-0.1, -0.4),
        fontsize=18,
        frameon=False,
        # prop={"weight": "bold"},
    )
    plt.title(title.title(), size=20, color="black", y=1.08, x=0.55, fontweight="bold")
    plt.plot()


def process_score(brage_score):
    return round(brage_score * 100, 2)


def get_and_group_scandeval(brage_agg_metrics_id="final", scandeval_date="09-10-2024"):
    agg_df = load_agg_metrics(brage_agg_metrics_id, group=True)
    models = get_models(agg_df)
    brage_full_desc, brage_keywords = brage_statistics(df=agg_df, models=models)

    def apply_brage(row):
        tmp_brage_full = brage_full_desc[brage_full_desc["models"] == row["model"]]
        tmp_brage_keyw = brage_keywords[brage_keywords["models"] == row["model"]]
        full_acc = tmp_brage_full["accuracy_mean"].values
        keyw_acc = tmp_brage_keyw["accuracy_mean"].values
        if len(full_acc) > 0:
            full_acc = process_score(full_acc[0])
        if len(keyw_acc) > 0:
            keyw_acc = process_score(keyw_acc[0])
        row["brage_full_desc"] = full_acc
        row["brage_keywords"] = keyw_acc
        return row

    sdval = load_scandeval(date=scandeval_date)
    combined_sdvals = []
    for _, dataset_name in scandeval_categories.items():
        _sdval = sdval[["model"] + [dataset_name]]
        _sdval = _sdval.apply(apply_brage, axis=1)
        _sdval = _sdval.sort_values(by="model")
        combined_sdvals.append(_sdval)

    sdvals = pd.concat(combined_sdvals)
    sdvals["mean_value"] = sdvals.drop("model", axis=1).apply(
        lambda x: float(x.split(" ± ")[0]), axis=1
    )
    sdvals = sdvals.groupby("model")["mean_value"].mean()
    cols = sdvals.columns.tolist()
    col_order = ["brage_full_desc", "brage_keywords"]
    other_cols = [col for col in cols if col not in col_order]
    cols = col_order + other_cols
    sdvals = sdvals[cols]
    sdvals = sdvals.sort_values(by="brage_full_desc", ascending=False)

    return sdvals
