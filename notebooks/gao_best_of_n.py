"""Explore alpacax Gao-inspired best-of-N experiments with marimo.

Run from repository root:
    uv run --extra train --with marimo --with matplotlib marimo edit notebooks/gao_best_of_n.py
"""
import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import csv
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt
    import marimo as mo
    import numpy as np

    return Path, csv, json, mo, np, plt


@app.cell
def _(mo):
    mo.md(
        r"""
        # Reward model overoptimization: best-of-\(N\)

        A small, inspectable companion to [Gao, Schulman & Hilton (2023)](https://arxiv.org/abs/2210.10760).

        **Question:** As we select more aggressively using an *imperfect proxy*, does the
        independent **gold evaluator** continue to improve?

        This notebook **reads saved data only**. No training or GPT-2 inference is
        triggered when a control changes. First run the synthetic example or the
        cached-text study to produce a `best_of_n.csv` file.
        """
    )
    return


@app.cell
def _(mo):
    path_input = mo.ui.text(
        value="runs/gao-sanity-42",
        label="Run directory (relative to repository root)",
        full_width=True,
    )
    x_axis = mo.ui.dropdown(
        options={"Number of candidates (N)": "n", "sqrt(KL) reference": "sqrt_kl"},
        value="sqrt_kl",
        label="Horizontal axis",
    )
    show_controls = mo.ui.checkbox(value=True, label="Include random and gold-oracle controls")
    mo.vstack([path_input, mo.hstack([x_axis, show_controls])])
    return path_input, x_axis, show_controls


@app.cell
def _(Path, csv, json, mo, path_input):
    directory = Path(path_input.value).expanduser()
    csv_path = directory / "best_of_n.csv"
    history_path = directory / "history.json"
    summary_path = directory / "summary.json"

    records: list[dict[str, float]] = []
    history: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    load_error = ""

    try:
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{csv_path} is missing. Run the training example first, or enter its output directory."
            )
        with csv_path.open(newline="", encoding="utf-8") as handle:
            records = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        if not records or "n" not in records[0] or "gold_mean" not in records[0]:
            raise ValueError("Expected best_of_n.csv with n and gold_mean columns")
        records.sort(key=lambda row: row["n"])
        if history_path.is_file():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        load_error = str(exc)

    mo.md(f"**Data:** {len(records)} rows from `{csv_path}`" if not load_error
          else f"**No usable results yet:** {load_error}")
    return directory, history, load_error, records, summary


@app.cell
def _(load_error, mo, np, plt, records, show_controls, x_axis):
    mo.stop(bool(load_error) or not records)
    x = np.array([r.get(x_axis.value, r["n"]) for r in records])
    n = np.array([r["n"] for r in records], dtype=int)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    if "proxy_mean" in records[0]:
        ax.plot(x, [r["proxy_mean"] for r in records], marker="o", label="Proxy-selected: proxy score")
    ax.plot(x, [r["gold_mean"] for r in records], marker="o", label="Proxy-selected: GOLD score")
    if show_controls.value:
        if "random_gold_mean" in records[0]:
            ax.plot(x, [r["random_gold_mean"] for r in records], linestyle="--", label="Random: gold")
        if "oracle_gold_mean" in records[0]:
            ax.plot(x, [r["oracle_gold_mean"] for r in records], linestyle="--", label="Gold oracle")
    ax.set_xlabel("sqrt(KL), theoretical BoN" if x_axis.value == "sqrt_kl" else "Best-of-N candidates")
    ax.set_ylabel("Mean reward (proxy and gold scales may differ)")
    ax.set_title("Proxy optimization versus gold performance")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    mo.as_html(fig)
    plt.close(fig)
    return n


@app.cell
def _(load_error, mo, np, plt, records, x_axis):
    mo.stop(bool(load_error) or not records)
    x = np.array([r.get(x_axis.value, r["n"]) for r in records])
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    if "selected_length_mean" in records[0]:
        ax2.plot(x, [r["selected_length_mean"] for r in records],
                 marker="o", label="Selected candidate length")
        ax2.set_ylabel("Mean length")
    else:
        ax2.text(0.5, 0.5, "No length data in this run", ha="center", va="center",
                 transform=ax2.transAxes)
    if "length_only_gold_mean" in records[0] and "selected_length_mean" in records[0]:
        twin = ax2.twinx()
        twin.plot(x, [r["length_only_gold_mean"] for r in records], linestyle=":",
                  label="Longest-candidate gold score")
        twin.set_ylabel("Gold score of longest candidate")
    ax2.set_xlabel("sqrt(KL)" if x_axis.value == "sqrt_kl" else "Best-of-N candidates")
    ax2.set_title("Length exploitation diagnostic")
    ax2.grid(alpha=0.25)
    fig2.tight_layout()
    mo.as_html(fig2)
    plt.close(fig2)
    return


@app.cell
def _(history, mo, plt):
    mo.stop(not history, mo.md("Training-loss history is not available for this run."))
    fig3, ax3 = plt.subplots(figsize=(9, 3.3))
    for split in ("train", "validation"):
        subset = [row for row in history if row.get("split") == split]
        if subset:
            ax3.plot([int(row["step"]) for row in subset],
                     [float(row["loss"]) for row in subset],
                     marker=".", label=split)
    ax3.set_xlabel("Completed optimizer updates")
    ax3.set_ylabel("Bradley–Terry loss")
    ax3.set_title("Training and held-out preference loss")
    ax3.legend()
    ax3.grid(alpha=0.25)
    fig3.tight_layout()
    mo.as_html(fig3)
    plt.close(fig3)
    return


@app.cell
def _(load_error, mo, records, summary):
    mo.stop(bool(load_error) or not records)
    first = records[0]
    best_gold = max(records, key=lambda row: row["gold_mean"])
    last = records[-1]
    changed = last["gold_mean"] - best_gold["gold_mean"]
    acc = summary.get("held_out_preference_accuracy")
    acc_line = f"**Held-out preference accuracy:** {acc:.1%}" if isinstance(acc, (int, float)) else ""
    mo.md(
        f"""
        ### What do these curves say?

        - Gold performance is highest at **N={int(best_gold['n'])}** in this run.
        - From that peak to N={int(last['n'])}, gold reward changes by **{changed:+.3f}**.
        - At N=1, gold reward is {first['gold_mean']:.3f}.
        - {acc_line}

        A declining gold score while the proxy score rises is evidence of
        **proxy overoptimization in this setup**, not a demonstration of Gao et al.'s
        quantitative scaling law. Compare the held-out gold oracle and random
        controls before interpreting the effect.
        """
    )
    mo.ui.table(records, selection=None, page_size=10)
    return


@app.cell
def _(mo, summary):
    if summary:
        mo.md("### Run metadata and caveats")
        mo.json(summary)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### How to extend the experiment

        1. Keep prompts split by **prompt ID** and candidate pools fixed across \(N\).
        2. Replace synthetic candidates with **cached frozen GPT-2 completions**.
        3. Train an **independent text-based Bradley–Terry proxy** on training
           prompt/completion preferences; never score test prompts for training.
        4. Compare a deterministic gold score, proxy-selected reward, gold-oracle,
           random baseline and length-only control.
        5. Repeat across two proxy-data sizes and multiple seeds before fitting
           any scaling-law coefficients.

        Reference: [Gao, Schulman & Hilton, *Scaling Laws for Reward Model
        Overoptimization* (arXiv:2210.10760)](https://arxiv.org/abs/2210.10760).
        """
    )
    return


if __name__ == "__main__":
    app.run()
