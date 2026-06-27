#!/usr/bin/env python3
"""TDOA localization and leave-one-out diagnostics."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy import optimize


def localize_least_squares(pos_xy: Dict[str, np.ndarray], tdoa_s: Dict[str, float], sigma_s: Optional[Dict[str, float]] = None, ref_unit: str = "five", c: float = 343.0, x0: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, float], float]:
    units = [u for u in tdoa_s if u != ref_unit and u in pos_xy]
    if len(units) < 2:
        raise ValueError("Need at least two non-reference TDOAs for 2D localization")
    pts = np.stack([pos_xy[u] for u in pos_xy])
    if x0 is None:
        x0 = np.mean(pts, axis=0)
    pref = np.asarray(pos_xy[ref_unit], dtype=float)
    def resid(x):
        x = np.asarray(x, dtype=float)
        dr = np.linalg.norm(x - pref)
        rr = []
        for u in units:
            pred = (np.linalg.norm(x - pos_xy[u]) - dr) / c
            s = float((sigma_s or {}).get(u, 0.001))
            rr.append((pred - float(tdoa_s[u])) / max(s, 1e-6))
        return np.asarray(rr)
    seeds = [x0, pref, *[pos_xy[u] for u in units]]
    best = None
    for seed in seeds:
        res = optimize.least_squares(resid, np.asarray(seed, dtype=float), loss="soft_l1", f_scale=2.0, max_nfev=2000)
        cost = float(np.sum(res.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, res.x)
    assert best is not None
    xy = np.asarray(best[1], dtype=float)
    pred = {}
    dr = np.linalg.norm(xy - pref)
    for u in units:
        pred[u] = float((np.linalg.norm(xy - pos_xy[u]) - dr) / c)
    resid_s = {u: float(tdoa_s[u] - pred[u]) for u in units}
    return xy, resid_s, float(best[0])


def leave_one_out_localizations(pos_xy: Dict[str, np.ndarray], tdoa_s: Dict[str, float], sigma_s: Optional[Dict[str, float]], ref_unit: str, c: float) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    units = [u for u in tdoa_s if u != ref_unit]
    for drop in units:
        sub = {u: t for u, t in tdoa_s.items() if u != drop}
        if len(sub) < 2:
            continue
        try:
            xy, _, _ = localize_least_squares(pos_xy, sub, sigma_s=sigma_s, ref_unit=ref_unit, c=c)
            out[drop] = xy
        except Exception:
            pass
    return out


def mc_location_samples(pos_xy: Dict[str, np.ndarray], tdoa_s: Dict[str, float], sigma_s: Dict[str, float], ref_unit: str, c: float, n: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    units = [u for u in tdoa_s if u != ref_unit]
    arr = []
    for _ in range(int(n)):
        samp = {u: float(rng.normal(tdoa_s[u], max(float(sigma_s.get(u, 0.001)), 1e-6))) for u in units}
        try:
            xy, _, _ = localize_least_squares(pos_xy, samp, sigma_s=sigma_s, ref_unit=ref_unit, c=c)
            if np.all(np.isfinite(xy)):
                arr.append(xy)
        except Exception:
            continue
    return np.asarray(arr, dtype=float)


def covariance_ellipse(samples_xy: np.ndarray, chi2_level: float = 5.991, n: int = 160) -> np.ndarray | None:
    if samples_xy.shape[0] < 5:
        return None
    ctr = np.mean(samples_xy, axis=0)
    cov = np.cov(samples_xy.T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 0)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.linspace(0, 2*math.pi, n, endpoint=False)
    circ = np.stack([np.cos(theta), np.sin(theta)])
    return (vecs @ (np.sqrt(vals * chi2_level)[:, None] * circ)).T + ctr
