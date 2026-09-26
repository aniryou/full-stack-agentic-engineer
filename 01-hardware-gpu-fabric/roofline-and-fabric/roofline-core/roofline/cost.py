"""The cost of a token: from $/GPU-hour to $/million tokens.

The one idea: a GPU-hour costs the same whether it produces a hundred tokens or
ten million, so

    $/M tokens = ($/GPU-hr x GPUs) / (tokens/s x 3600 x utilisation) x 1e6

Everything that raises tokens per GPU-second (batching, quantization, prefix
caching, a better engine) or keeps the GPUs busy (utilisation) cuts the cost.
Rent-vs-own is the same equation with a different $/GPU-hour and a utilisation
you commit to for years.
"""
from __future__ import annotations


def cost_per_million_tokens(price_per_gpu_hour: float, tokens_per_s: float,
                            utilisation: float = 1.0, n_gpus: int = 1) -> float:
    """$ per 1e6 tokens. tokens_per_s is what the n_gpus deliver together when busy."""
    return price_per_gpu_hour * n_gpus / (tokens_per_s * 3600 * utilisation) * 1e6


def utilisation(load_profile) -> float:
    """Capacity sized for the peak sits idle off-peak: utilisation = mean / max."""
    xs = list(load_profile)
    return sum(xs) / len(xs) / max(xs)


def owned_cost_per_hour(capex: float, years: float, power_kw: float, *, pue: float = 1.3,
                        usd_per_kwh: float = 0.10, opex_per_year: float = 0.0) -> tuple:
    """(fixed $/hr, energy $/busy-hr) for owned hardware.

    Fixed = straight-line depreciation + fixed opex (space, network, people, support).
    Energy = IT power x PUE x electricity price, paid (to first order) only while busy.
    """
    hours = years * 8760
    return capex / hours + opex_per_year / 8760, power_kw * pue * usd_per_kwh


def breakeven_utilisation(rent_per_hour: float, fixed_per_hour: float,
                          energy_per_hour: float = 0.0) -> float:
    """Utilisation above which owning beats renting the same capacity.

    Owning costs fixed + u x energy per wall-clock hour; renting only the hours you use
    costs u x rent. Equal at u = fixed / (rent - energy). > 1 means renting always wins.
    """
    if rent_per_hour <= energy_per_hour:
        return float("inf")
    return fixed_per_hour / (rent_per_hour - energy_per_hour)
