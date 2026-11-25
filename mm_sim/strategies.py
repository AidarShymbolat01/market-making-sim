"""Quoting strategies.

Every strategy has the same interface:

    bid, ask = strategy.quote(ref, inventory, t)

`ref` is the last fair value the market maker has seen, `inventory` is its
current position and `t` is the time step. All three work on numpy arrays
(one entry per simulated path). A quote of NaN means "do not quote that side".
"""
import numpy as np


class FixedSpread:
    """Quote symmetrically around the last price. No inventory control."""

    def __init__(self, half_spread):
        if half_spread <= 0:
            raise ValueError("half_spread must be positive")
        self.half_spread = half_spread

    def quote(self, ref, inventory, t):
        return ref - self.half_spread, ref + self.half_spread


class InventorySkew:
    """Shift both quotes against the current position and cap the position.

    Long inventory -> lower both quotes (sell more easily, buy less easily).
    Short inventory -> raise both quotes. Beyond +-max_inventory the strategy
    stops quoting the side that would increase the position.
    """

    def __init__(self, half_spread, skew, max_inventory):
        if half_spread <= 0:
            raise ValueError("half_spread must be positive")
        self.half_spread = half_spread
        self.skew = skew
        self.max_inventory = max_inventory

    def quote(self, ref, inventory, t):
        center = ref - self.skew * inventory
        bid = center - self.half_spread
        ask = center + self.half_spread
        bid = np.where(inventory >= self.max_inventory, np.nan, bid)
        ask = np.where(inventory <= -self.max_inventory, np.nan, ask)
        return bid, ask


class AvellanedaStoikov:
    """Stationary simplification of the Avellaneda-Stoikov (2008) quotes.

    reservation price  r = ref - inventory * gamma * sigma^2
    full spread        = gamma * sigma^2 + (2 / gamma) * ln(1 + gamma / k)

    The original paper has a finite horizon (T - t); here that factor is fixed
    to 1, which gives a constant spread and skew. `gamma` is risk aversion.
    """

    def __init__(self, gamma, k, sigma, max_inventory=None):
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.max_inventory = max_inventory
        self.half_spread = 0.5 * (
            gamma * sigma ** 2 + (2.0 / gamma) * np.log(1.0 + gamma / k)
        )

    def quote(self, ref, inventory, t):
        reservation = ref - inventory * self.gamma * self.sigma ** 2
        bid = reservation - self.half_spread
        ask = reservation + self.half_spread
        if self.max_inventory is not None:
            bid = np.where(inventory >= self.max_inventory, np.nan, bid)
            ask = np.where(inventory <= -self.max_inventory, np.nan, ask)
        return bid, ask
