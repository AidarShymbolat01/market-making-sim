"""Reproduce every figure and table in the README.

    python run_experiments.py

Writes PNG figures and results/summary.md into ./results. Everything is seeded,
so you get the same numbers every run.
"""
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mm_sim import (AvellanedaStoikov, FixedSpread, InventorySkew,
                    MarketParams, simulate, summarize)

OUT = "results"
BASE = MarketParams()

# Seeds are kept separate on purpose: strategy parameters were picked by hand
# while looking at SWEEP_SEED, and are then evaluated on the unseen EVAL_SEED.
SWEEP_SEED = 0
EVAL_SEED = 123


def spread_sweep():
    """Experiment 1: how wide should a fixed spread be?"""
    spreads = np.round(np.arange(0.1, 3.01, 0.2), 2)
    fractions = [0.0, 0.2, 0.4]
    mean_pnl = {}
    for f in fractions:
        params = replace(BASE, informed_frac=f)
        mean_pnl[f] = [
            summarize(simulate(FixedSpread(h), params, 400, SWEEP_SEED))["mean_pnl"]
            for h in spreads
        ]

    # Edge per trade, split by trader type, at the default informed share.
    edge_u, edge_i = [], []
    for h in spreads:
        s = summarize(simulate(FixedSpread(h), BASE, 400, SWEEP_SEED))
        edge_u.append(s["edge_per_uninformed_trade"])
        edge_i.append(s["edge_per_informed_trade"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for f in fractions:
        best = int(np.argmax(mean_pnl[f]))
        ax[0].plot(spreads, mean_pnl[f], marker="o", ms=3, label=f"{int(f * 100)}% informed")
        ax[0].scatter([spreads[best]], [mean_pnl[f][best]], s=80, marker="*", zorder=5)
    ax[0].set_xlabel("half-spread")
    ax[0].set_ylabel("mean final PnL")
    ax[0].set_title("More informed flow lowers PnL; the optimum is flat")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(spreads, edge_u, marker="o", ms=3, label="uninformed trades")
    ax[1].plot(spreads, edge_i, marker="o", ms=3, label="informed trades")
    ax[1].axhline(0, color="black", lw=0.8)
    ax[1].set_xlabel("half-spread")
    ax[1].set_ylabel("average edge per trade")
    ax[1].set_title("Where the money is made and lost (20% informed)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_spread_sweep.png"), dpi=140)
    plt.close(fig)

    best_by_fraction = {f: float(spreads[int(np.argmax(mean_pnl[f]))]) for f in fractions}
    return best_by_fraction


def compare_strategies(best_h):
    """Experiment 2: does inventory control help?"""
    strategies = {
        f"Fixed spread (h={best_h})": FixedSpread(best_h),
        "Inventory skew (h=1.0, skew=0.1)": InventorySkew(1.0, 0.1, 25),
        "Avellaneda-Stoikov (gamma=1)": AvellanedaStoikov(1.0, BASE.k, BASE.sigma, 25),
    }
    results = {name: simulate(s, BASE, 1000, EVAL_SEED) for name, s in strategies.items()}
    summaries = {name: summarize(r) for name, r in results.items()}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, r in results.items():
        ax[0].hist(r.pnl[:, -1], bins=40, alpha=0.55, label=name)
    ax[0].set_xlabel("final PnL")
    ax[0].set_ylabel("number of paths")
    ax[0].set_title("Distribution of final PnL (1000 paths)")
    ax[0].legend(fontsize=7)

    for name, r in results.items():
        ax[1].plot(r.inventory[0], lw=0.8, label=name)
    ax[1].axhline(0, color="black", lw=0.6)
    ax[1].set_xlabel("time step")
    ax[1].set_ylabel("inventory")
    ax[1].set_title("Inventory on one simulated path")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_strategies.png"), dpi=140)
    plt.close(fig)
    return summaries


def skew_frontier():
    """Experiment 3: the price of lower risk."""
    skews = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    points = []
    for sk in skews:
        s = summarize(simulate(InventorySkew(1.0, sk, 25), BASE, 600, SWEEP_SEED))
        points.append((s["std_pnl"], s["mean_pnl"]))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    xs, ys = zip(*points)
    ax.plot(xs, ys, marker="o")
    for sk, (x, y) in zip(skews, points):
        ax.annotate(f"{sk}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("std of final PnL (risk)")
    ax.set_ylabel("mean final PnL (reward)")
    ax.set_title("Skew trades reward for risk (labels = skew)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_skew_frontier.png"), dpi=140)
    plt.close(fig)


def write_summary(summaries):
    cols = [("mean_pnl", "mean PnL"), ("std_pnl", "std PnL"), ("p5_pnl", "5th pct PnL"),
            ("mean_max_drawdown", "avg max drawdown"), ("mean_abs_inventory", "avg |inventory|"),
            ("max_abs_inventory", "max |inventory|"), ("trades_per_path", "trades/path")]
    lines = ["| Strategy | " + " | ".join(c[1] for c in cols) + " |",
             "|---|" + "---:|" * len(cols)]
    for name, s in summaries.items():
        lines.append(f"| {name} | " + " | ".join(f"{s[c[0]]:.1f}" for c in cols) + " |")
    text = "\n".join(lines)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write(text + "\n")
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    best = spread_sweep()
    print("Best fixed half-spread by informed share:", best)
    summaries = compare_strategies(best[0.2])
    skew_frontier()
    print(write_summary(summaries))
    print("\nEdge per trade (strategy comparison):")
    for name, s in summaries.items():
        print(f"  {name:36s} uninformed {s['edge_per_uninformed_trade']:+.2f}   "
              f"informed {s['edge_per_informed_trade']:+.2f}   "
              f"informed share {s['informed_trade_share']:.1%}")


if __name__ == "__main__":
    main()
