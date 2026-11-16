from __future__ import annotations


def speedup(t_sequential: float, t_parallel: float) -> float:
    """S = T_secuencial / T_paralelo."""
    if t_sequential <= 0 or t_parallel <= 0:
        raise ValueError("Los tiempos deben ser positivos")
    return t_sequential / t_parallel


def efficiency(t_sequential: float, t_parallel: float, n_workers: int) -> float:
    """E = S / p."""
    if n_workers < 1:
        raise ValueError("n_workers debe ser al menos 1")
    return speedup(t_sequential, t_parallel) / n_workers


def amdahl_speedup(parallel_fraction: float, n_workers: int) -> float:
    """Límite teórico de Amdahl: S(p) = 1 / ((1 - f) + f / p)."""
    if not 0 <= parallel_fraction <= 1:
        raise ValueError("parallel_fraction debe estar entre 0 y 1")
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / n_workers)
