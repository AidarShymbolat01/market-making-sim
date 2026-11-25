"""Summary statistics for a SimResult."""
import numpy as np


def summarize(result):
    """Return a dict of headline numbers (averaged across simulated paths)."""
    final = result.pnl[:, -1]
    running_max = np.maximum.accumulate(result.pnl, axis=1)
    max_drawdown = (running_max - result.pnl).max(axis=1)

    trades = result.n_trades.sum()
    informed = result.n_informed_trades.sum()
    uninformed = trades - informed

    std = final.std(ddof=1)
    return {
        "mean_pnl": final.mean(),
        "std_pnl": std,
        "p5_pnl": np.percentile(final, 5),
        "mean_over_std": final.mean() / std if std > 0 else float("nan"),
        "mean_max_drawdown": max_drawdown.mean(),
        "mean_abs_inventory": np.abs(result.inventory).mean(),
        "max_abs_inventory": np.abs(result.inventory).max(),
        "trades_per_path": result.n_trades.mean(),
        "informed_trade_share": informed / trades if trades else 0.0,
        "edge_per_uninformed_trade": result.edge_uninformed.sum() / max(uninformed, 1),
        "edge_per_informed_trade": result.edge_informed.sum() / max(informed, 1),
    }
