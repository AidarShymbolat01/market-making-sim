"""Market parameters shared by the simulator and the experiments."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketParams:
    start_price: float = 100.0   # initial fair value
    sigma: float = 0.5           # std-dev of the fair-value move each step
    steps: int = 2000            # time steps per simulated path
    arrival_prob: float = 0.8    # probability that a trader shows up in a step
    informed_frac: float = 0.2   # share of traders who know the new fair value
    k: float = 1.0               # price sensitivity of uninformed traders
