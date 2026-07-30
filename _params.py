"""Parameter space for evolution — 7 parameters (≤7 to avoid overfitting).

V2 defaults are included as anchor. Evolution explores around these values.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class StrategyParams:
    """7 parameters that define a SniperTrend strategy variant."""
    entry_sma_mult: float    # SMA ratio for SHORT entry (V2: 0.999)
    entry_rsi_min: int       # RSI minimum for entry (V2: 30)
    entry_rsi_max: int       # RSI maximum for entry (V2: 55)
    take_profit_pct: float   # Take-profit % (V2: 0.0, V10: 0.015)
    hold_ticks: int          # Min hold time in ticks (V2: 6)
    exit_sma_mult: float     # SMA ratio for exit (V2: 1.003)
    position_size: float     # Position size fraction (V2: 0.3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_sma_mult': round(self.entry_sma_mult, 4),
            'entry_rsi_min': int(self.entry_rsi_min),
            'entry_rsi_max': int(self.entry_rsi_max),
            'take_profit_pct': round(self.take_profit_pct, 4),
            'hold_ticks': int(self.hold_ticks),
            'exit_sma_mult': round(self.exit_sma_mult, 4),
            'position_size': round(self.position_size, 3),
        }


# Parameter ranges for mutation/crossover
PARAM_RANGES = {
    'entry_sma_mult':  (0.995, 1.005),
    'entry_rsi_min':   (20, 40),
    'entry_rsi_max':   (45, 60),
    'take_profit_pct': (0.005, 0.025),
    'hold_ticks':      (6, 30),
    'exit_sma_mult':   (1.002, 1.005),
    'position_size':   (0.2, 0.4),
}

# V2 reference values — the anchor
V2_PARAMS = StrategyParams(
    entry_sma_mult=0.999,
    entry_rsi_min=30,
    entry_rsi_max=55,
    take_profit_pct=0.0,
    hold_ticks=6,
    exit_sma_mult=1.003,
    position_size=0.3,
)


def random_params() -> StrategyParams:
    """Generate random parameters within ranges."""
    return StrategyParams(
        entry_sma_mult=random.uniform(*PARAM_RANGES['entry_sma_mult']),
        entry_rsi_min=random.randint(*PARAM_RANGES['entry_rsi_min']),
        entry_rsi_max=random.randint(*PARAM_RANGES['entry_rsi_max']),
        take_profit_pct=random.uniform(*PARAM_RANGES['take_profit_pct']),
        hold_ticks=random.randint(*PARAM_RANGES['hold_ticks']),
        exit_sma_mult=random.uniform(*PARAM_RANGES['exit_sma_mult']),
        position_size=random.uniform(*PARAM_RANGES['position_size']),
    )


def mutate(params: StrategyParams, rate: float = 0.05) -> StrategyParams:
    """Mutate parameters with given rate (0.05 = 5% chance per param)."""
    d = params.to_dict()
    for key, (lo, hi) in PARAM_RANGES.items():
        if random.random() < rate:
            if isinstance(lo, int) and isinstance(hi, int):
                d[key] = random.randint(lo, hi)
            else:
                # Gaussian mutation around current value
                current = d[key]
                sigma = (hi - lo) * 0.1  # 10% of range
                new_val = current + random.gauss(0, sigma)
                d[key] = max(lo, min(hi, new_val))
                if isinstance(lo, int):
                    d[key] = int(d[key])
    return StrategyParams(**d)


def crossover(parent1: StrategyParams, parent2: StrategyParams) -> StrategyParams:
    """Uniform crossover — each gene randomly from either parent."""
    d1 = parent1.to_dict()
    d2 = parent2.to_dict()
    child = {}
    for key in d1:
        child[key] = d1[key] if random.random() < 0.5 else d2[key]
    return StrategyParams(**child)


def tournament_select(population: list, fitnesses: list, k: int = 3) -> StrategyParams:
    """Tournament selection — pick best of k random individuals."""
    indices = random.sample(range(len(population)), min(k, len(population)))
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]
