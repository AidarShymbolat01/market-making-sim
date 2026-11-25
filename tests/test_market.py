import unittest
from dataclasses import replace

import numpy as np

from mm_sim import (AvellanedaStoikov, FixedSpread, InventorySkew,
                    MarketParams, simulate, summarize)

SHORT = MarketParams(steps=400)


class SimulationTests(unittest.TestCase):
    def test_pnl_identity(self):
        """Final PnL must equal edge from trades plus PnL from price moves."""
        r = simulate(InventorySkew(0.8, 0.05, 20), SHORT, n_paths=50, seed=1)
        total = r.edge_uninformed + r.edge_informed + r.inventory_pnl
        np.testing.assert_allclose(r.pnl[:, -1], total, atol=1e-8)

    def test_inventory_cap_is_respected(self):
        r = simulate(InventorySkew(0.3, 0.0, 5), SHORT, n_paths=100, seed=2)
        self.assertLessEqual(np.abs(r.inventory).max(), 5)
        self.assertEqual(np.abs(r.inventory).max(), 5)  # the cap does bind

    def test_no_traders_means_no_trades_and_zero_pnl(self):
        params = replace(SHORT, arrival_prob=0.0)
        r = simulate(FixedSpread(1.0), params, n_paths=20, seed=3)
        self.assertEqual(r.n_trades.sum(), 0)
        self.assertTrue(np.all(r.pnl == 0))

    def test_zero_volatility_never_loses(self):
        """If the price never moves, spread capture is the only PnL source."""
        params = replace(SHORT, sigma=0.0)
        r = simulate(FixedSpread(0.5), params, n_paths=50, seed=4)
        self.assertEqual(r.n_informed_trades.sum(), 0)
        self.assertTrue(np.all(r.pnl[:, -1] >= 0))

    def test_same_seed_is_reproducible(self):
        a = simulate(FixedSpread(1.0), SHORT, n_paths=30, seed=7)
        b = simulate(FixedSpread(1.0), SHORT, n_paths=30, seed=7)
        c = simulate(FixedSpread(1.0), SHORT, n_paths=30, seed=8)
        np.testing.assert_array_equal(a.pnl, b.pnl)
        self.assertFalse(np.array_equal(a.pnl, c.pnl))

    def test_wider_spread_means_fewer_trades(self):
        tight = simulate(FixedSpread(0.3), SHORT, n_paths=100, seed=5)
        wide = simulate(FixedSpread(2.0), SHORT, n_paths=100, seed=5)
        self.assertGreater(tight.n_trades.mean(), wide.n_trades.mean())

    def test_skew_reduces_inventory_risk(self):
        plain = summarize(simulate(FixedSpread(1.0), SHORT, n_paths=200, seed=6))
        skewed = summarize(simulate(InventorySkew(1.0, 0.1, 25), SHORT, n_paths=200, seed=6))
        self.assertLess(skewed["mean_abs_inventory"], plain["mean_abs_inventory"])


class StrategyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.ref = 100 + rng.normal(size=1000)
        self.inv = rng.integers(-30, 31, size=1000).astype(float)

    def test_quotes_never_cross(self):
        strategies = [FixedSpread(0.5), InventorySkew(0.5, 0.2, 10),
                      AvellanedaStoikov(1.0, 1.0, 0.5, max_inventory=10)]
        for s in strategies:
            bid, ask = s.quote(self.ref, self.inv, 0)
            both = ~np.isnan(bid) & ~np.isnan(ask)
            self.assertTrue(np.all(bid[both] < ask[both]), type(s).__name__)

    def test_skew_moves_quotes_against_inventory(self):
        s = InventorySkew(1.0, 0.1, 100)
        bid_long, ask_long = s.quote(np.array([100.0]), np.array([10.0]), 0)
        bid_flat, ask_flat = s.quote(np.array([100.0]), np.array([0.0]), 0)
        self.assertLess(ask_long[0], ask_flat[0])  # keen to sell when long
        self.assertLess(bid_long[0], bid_flat[0])  # reluctant to buy when long

    def test_side_switches_off_at_inventory_limit(self):
        s = InventorySkew(1.0, 0.0, 5)
        bid, ask = s.quote(np.array([100.0, 100.0]), np.array([5.0, -5.0]), 0)
        self.assertTrue(np.isnan(bid[0]) and not np.isnan(ask[0]))
        self.assertTrue(np.isnan(ask[1]) and not np.isnan(bid[1]))

    def test_avellaneda_stoikov_half_spread_formula(self):
        gamma, k, sigma = 2.0, 1.5, 0.4
        expected = 0.5 * (gamma * sigma ** 2 + (2 / gamma) * np.log(1 + gamma / k))
        self.assertAlmostEqual(AvellanedaStoikov(gamma, k, sigma).half_spread, expected)

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            FixedSpread(0)
        with self.assertRaises(ValueError):
            InventorySkew(-1, 0.1, 5)
        with self.assertRaises(ValueError):
            AvellanedaStoikov(0, 1.0, 0.5)


if __name__ == "__main__":
    unittest.main()
