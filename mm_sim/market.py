"""Simulation of one market maker trading against a stream of random traders.

Model (one step = one tick of time):

1. The market maker posts a bid and an ask based on the last fair value it saw.
2. The true fair value then moves:  V_t = V_{t-1} + sigma * N(0, 1).
3. With probability `arrival_prob` one trader arrives:
   * informed trader (prob `informed_frac`): already knows V_t. Buys if the
     ask is below V_t, sells if the bid is above V_t, otherwise stays away.
   * uninformed trader: picks buy or sell at random and trades with
     probability min(1, exp(-k * d)), where d is how far the quote is above
     (for a buy) or below (for a sell) the true value. Wider quotes -> fewer
     trades.
4. The market maker sees V_t and uses it as its new reference price.

Because the market maker always quotes around the *old* price, informed
traders can pick it off after a big move. That is adverse selection: it
punishes quotes that are too tight, while quotes that are too wide lose
the uninformed flow.

All paths are simulated at once with numpy arrays of shape (n_paths,).
"""
from dataclasses import dataclass

import numpy as np

from .params import MarketParams


@dataclass
class SimResult:
    pnl: np.ndarray                # (n_paths, steps) mark-to-market PnL
    inventory: np.ndarray          # (n_paths, steps) position after each step
    n_trades: np.ndarray           # (n_paths,)
    n_informed_trades: np.ndarray  # (n_paths,)
    edge_uninformed: np.ndarray    # (n_paths,) sum of (price - fair value) edge
    edge_informed: np.ndarray      # (n_paths,)
    inventory_pnl: np.ndarray      # (n_paths,) PnL from price moves on inventory


def simulate(strategy, params=MarketParams(), n_paths=500, seed=0):
    """Run `n_paths` independent paths and return a SimResult."""
    rng = np.random.default_rng(seed)
    n, steps = n_paths, params.steps

    ref = np.full(n, params.start_price)
    cash = np.zeros(n)
    inv = np.zeros(n)

    pnl = np.empty((n, steps))
    inv_hist = np.empty((n, steps))
    n_trades = np.zeros(n)
    n_informed_trades = np.zeros(n)
    edge_uninformed = np.zeros(n)
    edge_informed = np.zeros(n)
    inventory_pnl = np.zeros(n)

    for t in range(steps):
        bid, ask = strategy.quote(ref, inv, t)
        value = ref + params.sigma * rng.standard_normal(n)

        arrives = rng.random(n) < params.arrival_prob
        informed = rng.random(n) < params.informed_frac
        wants_to_buy = rng.random(n) < 0.5
        u = rng.random(n)

        # NaN quotes (side switched off) compare as False, so no trade happens.
        with np.errstate(invalid="ignore", over="ignore"):
            p_buy = np.minimum(1.0, np.exp(-params.k * (ask - value)))
            p_sell = np.minimum(1.0, np.exp(-params.k * (value - bid)))
            trader_buys = np.where(informed, value > ask, wants_to_buy & (u < p_buy))
            trader_sells = np.where(informed, value < bid, ~wants_to_buy & (u < p_sell))
        trader_buys = trader_buys & arrives
        trader_sells = trader_sells & arrives

        # A buying trader lifts our ask (we sell); a selling trader hits our bid.
        we_sell = trader_buys
        we_buy = trader_sells
        traded = we_sell | we_buy

        # PnL from the price move on the inventory we held *before* this trade.
        inventory_pnl += inv * (value - ref)

        # Edge = how far from the true value we traded, in our favour if > 0.
        edge = np.where(we_sell, ask - value, 0.0) + np.where(we_buy, value - bid, 0.0)
        edge_uninformed += np.where(informed, 0.0, edge)
        edge_informed += np.where(informed, edge, 0.0)

        cash += np.where(we_sell, ask, 0.0) - np.where(we_buy, bid, 0.0)
        inv = inv - we_sell.astype(float) + we_buy.astype(float)

        n_trades += traded
        n_informed_trades += traded & informed

        pnl[:, t] = cash + inv * value
        inv_hist[:, t] = inv
        ref = value

    return SimResult(pnl, inv_hist, n_trades, n_informed_trades,
                     edge_uninformed, edge_informed, inventory_pnl)
