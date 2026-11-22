"""Build M2-v5 defense figures from committed aggregate reports."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from medpredict_ml.common import read_json

REPORTS = Path(__file__).resolve().parents[1] / "reports"
BINS = {"lt110": "<110", "110_to_lt130": "110–<130", "130_to_lt150": "130–<150", "ge150": "≥150"}
RISK = {
    "bpq020_no_ge150": "BPQ020 = нет · САД ≥150",
    "bpq020_no_ge140": "BPQ020 = нет · САД ≥140",
    "medication_no_ge150": "Препараты = нет · САД ≥150",
    "medication_no_ge140": "Препараты = нет · САД ≥140",
}
POINT = {"v2": ("Опубликованная v2", "#556478", "o"),
         "main": ("Ridge · основной", "#2457A7", "s"),
         "safe": ("MLP · вторичный", "#00766B", "D")}
PAPER, INK, GRID = "#F9FAFC", "#1A2939", "#DCE3EB"


def figure(rows, cols, size, title, note, *, left=.1, bottom=.16, top=.85):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=left, right=.96, bottom=bottom, top=top, hspace=.58, wspace=.36)
    fig.text(.07, .95, title, size=19, weight="bold", color=INK, va="top")
    fig.text(.07, .045, note, size=9, color="#425164", va="bottom", linespacing=1.45)
    return fig, axes


def tidy(ax, axis="y"):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0, labelcolor=INK)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, lw=.8)


def signed_error(metrics):
    fig, axes = figure(1, 1, (10, 4.8), "Смещение прогноза по измеренному САД",
                       "NHANES 2015–2016 · единая fasting-когорта для трёх моделей.\n"
                       "Точки показывают взвешенные средние САД и ошибки в диапазоне. "
                       "Отрицательное смещение означает занижение.",
                       bottom=.23)
    ax = axes[0, 0]
    tidy(ax)
    rows = [metrics["defense_point"]["sbp_" + key] for key in BINS]
    centers = np.array([row["weighted_mean_sbp"] for row in rows], dtype=float)
    for name, (label, color, marker) in POINT.items():
        ax.plot(centers, [row["models"][name].get("weighted_mean_signed_error") for row in rows],
                marker=marker, color=color, lw=2.3, ms=6, label=label)
    ax.axhline(0, color="#758397", lw=1)
    shown = np.isfinite(centers)
    labels = [f"{label}\n(n={row['n']})" for label, row in zip(BINS.values(), rows, strict=True)]
    ax.set_xticks(centers[shown], [label for label, ok in zip(labels, shown, strict=True) if ok])
    ax.set_xlabel("Измеренное САД, мм рт. ст.", labelpad=7)
    ax.set_ylabel("Взвешенное смещение, мм рт. ст.")
    ax.legend(loc="lower left", frameon=False, ncol=1)
    return fig


def experiment_path(cv, grid, freeze, metrics):
    fig, axes = figure(1, 2, (13, 5.9), "Путь от проверки признаков до внешней оценки",
                       "C · Разности внутри каждой пары. Сравнения используют разные когорты.\n"
                       "D · Тот же OOF-состав для двух кандидатов. E · Замороженные модели против v2 на парных строках. "
                       "Интервалы E — парные 95% PSU-ДИ.", left=.23, bottom=.17)
    left, right = axes[0]
    comparison = {
        "C1 · HbA1c": cv["C1"], "C2 · Расширение 2017–2020": cv["C2"],
        "C3 · Добавление 2021–2023": cv["C3"], "C4 · Препараты": cv["C4"]["medication"],
        "C4 · BPQ020": cv["C4"]["bpq020"], "C4 · Курение": cv["C4"]["smoking"],
        "C4 · PAD680": cv["C4"]["pad680"],
    }
    for ax in (left, right):
        tidy(ax, "x")
        ax.invert_yaxis()
    changes = [item["delta_paired"]["high_mae"]["mean"] for item in comparison.values()]
    left.barh(range(len(changes)), changes, color=["#00766B" if value < 0 else "#AB5636" for value in changes], height=.53)
    left.axvline(0, color="#758397", lw=1)
    left.set_yticks(range(len(changes)), list(comparison))
    left.set_xlim(min(changes) - .55, max(changes) + .55)
    left.set_title("C1–C4 · MAE при САД ≥150", loc="left", weight="bold")
    left.set_xlabel("Кандидат минус основа, мм рт. ст.")
    for y, value in enumerate(changes):
        left.annotate(f"{value:+.2f}", (value, y), xytext=(4 if value >= 0 else -4, 0),
                      textcoords="offset points", ha="left" if value >= 0 else "right", va="center", fontsize=9)

    bootstrap = metrics["paired_psu_bootstrap"]["metrics"]
    tail = [item["point_paired"]["sbp_ge150"]["models"] for item in metrics["candidates"]]
    d_values = [grid["selected"]["high_reduction_percent"], freeze["safe_point"]["high_reduction_percent"]]
    e_values = [100 * (1 - row[name]["weighted_mae"] / row["v2"]["weighted_mae"])
                if row[name]["weighted_mae"] is not None and row["v2"]["weighted_mae"] else np.nan
                for name, row in zip(("main", "safe"), tail, strict=True)]
    for i, (name, score) in enumerate(zip(("main", "safe", "main", "safe"), d_values + e_values, strict=True)):
        _, color, marker = POINT[name]
        if i >= 2 and (ci := bootstrap[f"{name}_ge150_reduction"]["ci95"]) is not None:
            right.hlines(i, 100 * ci[0], 100 * ci[1], color=color, lw=2)
        if np.isfinite(score):
            right.plot(score, i, marker, color=color, ms=7)
            right.annotate(f"{score:.1f}%", (score, i), xytext=(0, -10), textcoords="offset points",
                           ha="center", va="top", fontsize=9)
    right.axvline(0, color="#758397", lw=1)
    right.vlines(25, 1.65, 3.45, color="#AB5636", linestyle="--")
    right.annotate("Порог E", (25, 1.65), xytext=(0, 4), textcoords="offset points", ha="center", color="#83432E", fontsize=9)
    right.set_yticks(range(4), ["D · Ridge", "D · MLP", "E · Ridge", "E · MLP"])
    right.set_ylim(3.6, -.6)
    right.set_title("D и E · Снижение MAE при САД ≥150", loc="left", weight="bold")
    right.set_xlabel("Относительное снижение, %")
    return fig


def intervals(candidates):
    fig, axes = figure(2, 2, (12, 7.2), "CQR и симметричный интервал v2",
                       "NHANES 2015–2016 · когорты спарены с v2 отдельно для каждого кандидата.\n"
                       "Показаны невзвешенные доли и средние. Пороги внедрения оцениваются на расширенной когорте.",
                       top=.83)
    x = range(5)
    groups = {"overall": "Все", **BINS}
    for col, item in enumerate(candidates):
        cqr, v2 = (item[key] for key in ("cqr_paired", "v2_interval_paired"))
        name, color, marker = POINT[item["candidate"]]
        for row, (metric, unit, scale) in enumerate((("coverage", "Покрытие, %", 100), ("width", "Ширина, мм рт. ст.", 1))):
            ax = axes[row, col]
            tidy(ax)
            for data, label, rgb, mark in ((v2, "Симметричный v2", POINT["v2"][1], "o"), (cqr, "CQR", color, marker)):
                series = np.asarray([data[key][metric] for key in groups], dtype=float)
                ax.plot(x, series * scale, marker=mark, color=rgb, lw=2, label=label)
            if row == 0:
                ax.set_ylim(-3, 104)
                ax.set_title(name, loc="left", weight="bold")
                ax.legend(frameon=False, loc="lower left", fontsize=9)
            ax.set_ylabel(unit)
            ax.set_xticks(x, [f"{label}\n(n={cqr[key]['n']})" for key, label in groups.items()])
    align_limits(axes[1], "ylim")
    return fig


def align_limits(axes, axis):
    ranges = [getattr(ax, f"get_{axis}")() for ax in axes]
    limits = min(a for a, _ in ranges), max(b for _, b in ranges)
    for ax in axes:
        getattr(ax, f"set_{axis}")(*limits)


def probabilities(metrics):
    probability = metrics["probability"]
    models = {
        "hist_gradient_boosting": ("HistGradientBoosting", "#2457A7"),
        "logistic_same_inputs": ("Logistic · 6 входов", "#00766B"),
        "logistic_age_only": ("Logistic · только возраст", "#AB5636"),
        "constant": ("Константа", "#556478"),
    }
    first = probability["hist_gradient_boosting"]
    delta_age = metrics["probability_auc_deltas"]["interface_logistic_minus_age_logistic"]
    delta_note = "" if delta_age is None else f"\nΔAUC Logistic на 6 входах к Logistic по возрасту = {delta_age:+.3f}."
    fig, axes = figure(1, 3, (13, 4.1), "Вероятность измеренного САД ≥140",
                       f"NHANES 2015–2016 · расширенная когорта n={first['n']}, событий {first['event_n']}.\n"
                       "Взвешенные метрики. Горизонтальные отрезки на ROC-AUC — 95% PSU-ДИ." + delta_note,
                       left=.20, bottom=.24, top=.78)
    bootstrap = metrics["paired_psu_bootstrap"]["metrics"]
    for j, (ax, (key, title)) in enumerate(zip(axes[0], (("roc_auc", "ROC-AUC ↑"),
            ("pr_auc_average_precision", "PR-AUC ↑"), ("brier", "Brier ↓")), strict=True)):
        tidy(ax, "x")
        ax.invert_yaxis()
        ax.set_yticks(range(4), [label for label, _ in models.values()] if j == 0 else [""] * 4)
        ax.set_ylim(3.6, -.6)
        ax.set_title(title, loc="left", weight="bold")
        for i, (model, (_, color)) in enumerate(models.items()):
            value = probability[model][key]
            if value is None:
                continue
            if key == "roc_auc" and (ci := bootstrap.get(f"probability_{model}_auc", {}).get("ci95")) is not None:
                ax.hlines(i, *ci, color=color, lw=2)
            ax.plot(value, i, "o", color=color, ms=7)
            ax.annotate(f"{value:.3f}", (value, i), xytext=(0, -10), textcoords="offset points",
                        ha="center", va="top", fontsize=9)
        ax.margins(x=.24)
    return fig


def undiagnosed_risk(candidates):
    fig, axes = figure(2, 2, (12.5, 7.4), "Ошибки в срезах недиагностированного риска",
                       "NHANES 2015–2016 · ошибки взвешены fasting-весами. В каждом ряду все три модели сравниваются на одних строках.\n"
                       "Пятипризнаковая основа соответствует семейству кандидата. Смещение равно прогнозу минус измеренное САД.\n"
                       "* Срез с n<30 описательный.",
                       left=.28, top=.82)
    for i, item in enumerate(candidates):
        name = item["candidate"]
        label, color, marker = POINT[name]
        paired = item["point_paired"]
        labels = [f"{label} (n={paired[group]['n']})" + ("*" if paired[group]["n"] < 30 else "")
                  for group, label in RISK.items()]
        for j, (metric, title) in enumerate((("weighted_mae", "MAE"), ("weighted_mean_signed_error", "Знаковое смещение"))):
            ax = axes[i, j]
            tidy(ax, "x")
            ax.set_yticks(range(4), labels if j == 0 else [""] * 4, fontsize=9)
            ax.set_ylim(3.4, -.4)
            ax.set_title(f"{label} · {title}", loc="left", weight="bold", size=11)
            ax.set_xlabel("мм рт. ст.")
            series = [("v2", "v2", POINT["v2"][1], "o"), (name + "_base_5f", "Основа 5 полей", "#AB5636", "^"),
                      (name, "Кандидат", color, marker)]
            for k, (model, title, rgb, mark) in enumerate(series):
                values = np.array([paired[group]["models"][model][metric] for group in RISK], dtype=float)
                ax.scatter(values, np.arange(4) + (k - 1) * .16, color=rgb, marker=mark, label=title, s=34)
            if i == 0 and j == 0:
                ax.legend(frameon=False, ncol=3, bbox_to_anchor=(0, 1.14), loc="lower left", fontsize=9)
    for column in axes.T:
        align_limits(column, "xlim")
    return fig


def make_figures(cv, grid, freeze, metrics, destination):
    figures = {
        "sbp_error": signed_error(metrics),
        "experiment_path": experiment_path(cv, grid, freeze, metrics),
        "intervals": intervals(metrics["candidates"]),
        "probability": probabilities(metrics),
        "undiagnosed_risk": undiagnosed_risk(metrics["candidates"]),
    }
    for name, fig in figures.items():
        fig.savefig(destination / f"m2_v5_{name}.png", dpi=160, facecolor=PAPER)
        plt.close(fig)


if __name__ == "__main__":
    evaluation = read_json(REPORTS / "m2_v5_e_final.json")
    if evaluation["non_confirmatory"] or evaluation["cycle"] != "2015-2016":
        raise ValueError("Defense figures require the final external evaluation")
    make_figures(read_json(REPORTS / "m2_v5_cv.json")["stages"], read_json(REPORTS / "m2_v5_grid.json"),
                 read_json(REPORTS / "m2_v5_frozen.json")["pre_e"], evaluation["metrics"], REPORTS / "figures")
