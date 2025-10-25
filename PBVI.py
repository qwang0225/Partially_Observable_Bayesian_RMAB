import numpy as np
from utils import *

# Caches for faster PBVI Whittle computation
_LOCAL_MODEL_CACHE = {}
_V_CACHE = {}
_LAST_LAMBDA = None


def _q(x, eps=1e-6):
    """Quantize a float to a coarse grid for stable cache keys."""
    return float(np.round(float(x) / eps) * eps)


def get_or_build_local_model(p, th0, th1, P01, P11, rng,
                             n_rollouts=8, depth=3, active_reveals_H=True,
                             jitter=1e-6):
    """Build or fetch a small local belief set and its kernels near p.

    Returns B (sorted array), P0 (passive kernel), P1 (active kernel).
    Uses a cache keyed by rounded inputs to avoid re-building.
    """
    key = (
        _q(p, 1e-5), _q(th0, 1e-5), _q(th1, 1e-5),
        tuple(np.round(np.asarray(P01).ravel(), 6)),
        tuple(np.round(np.asarray(P11).ravel(), 6)),
        int(n_rollouts), int(depth), bool(active_reveals_H)
    )
    if key in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[key]

    # Build local B similarly to smc_value_iter_local
    B = [0.0, 1.0]
    if p not in B:
        B.append(float(p))
    B = np.array(sorted(B))
    
    for _ in range(n_rollouts):
        b = float(p)
        for _d in range(depth):
            pt = predict_belief(b, a=0, P01=P01, P11=P11)
            py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
            for cand in (p0_next, p1_next):
                cand = float(np.clip(cand, 0.0, 1.0))
                if np.min(np.abs(B - cand)) > 1e-6:
                    B = np.sort(np.append(B, cand + rng.uniform(-jitter, jitter)))
            if active_reveals_H and rng.random() < 0.3:
                # Include active collapse endpoints already captured by {0,1}
                pass
            b = p1_next if rng.random() < py1 else p0_next

    B = np.clip(B, 0.0, 1.0)
    B = np.unique(np.round(B, 8))
    S = len(B)

    # Active kernel
    P_active = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=1, P01=P01, P11=P11)
        for j, wj in split_to_grid(0.0, B):
            P_active[i, j] += (1.0 - pt) * wj
        for j, wj in split_to_grid(1.0, B):
            P_active[i, j] += pt * wj

    # Passive kernel
    P_passive = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=0, P01=P01, P11=P11)
        py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
        for j, wj in split_to_grid(p0_next, B):
            P_passive[i, j] += py0 * wj
        for j, wj in split_to_grid(p1_next, B):
            P_passive[i, j] += py1 * wj

    _LOCAL_MODEL_CACHE[key] = (B, P_passive, P_active)
    return B, P_passive, P_active


def value_iter_with_warm_start(B, P0, P1, lam, gamma, V0=None, vi_tol=1e-6, max_iter=500):
    """Run VI on (B,P0,P1) with reward r0=B, r1=B-lam, optionally warm-starting V."""
    S = len(B)
    r0 = B.copy()
    r1 = B.copy() - float(lam)
    V = V0.copy() if V0 is not None else np.zeros(S)
    for _ in range(max_iter):
        V_old = V.copy()
        Q0 = r0 + gamma * (P0 @ V_old)
        Q1 = r1 + gamma * (P1 @ V_old)
        V = np.maximum(Q0, Q1)
        if np.max(np.abs(V - V_old)) < vi_tol:
            break
    return V, Q0, Q1

def _nn_index(x, pts):
    # nearest neighbor index in pts
    return int(np.argmin(np.abs(pts - x)))

def smc_value_iter_local(th0, th1, p, transition_matrix=None, lamb_val=0.0,
                         discount=0.95, vi_tol=1e-6, max_iter=1000,
                         n_rollouts=8, depth=3, jitter=1e-6,
                         P01=None, P11=None):
    """
    Value iteration with a SMALL point set of reachable beliefs (no global grid).
    Returns: Q0_at_p, Q1_at_p, greedy_action, belief_points (for inspection)
    """
    assert 0.0 <= th0 <= 1.0 and 0.0 <= th1 <= 1.0
    assert 0.0 <= p <= 1.0
    assert 0.0 < discount < 1.0

    # Allow either full transition_matrix or direct P01/P11 inputs
    if transition_matrix is not None:
        P01 = transition_matrix[0, :, 1]  # shape (2,)
        P11 = transition_matrix[1, :, 1]  # shape (2,)
    else:
        assert P01 is not None and P11 is not None, "Provide P01,P11 or transition_matrix"

    # ---- 1) Build a tiny local belief set B starting from {p, 0, 1} ----
    B = [0.0, 1.0]
    if p not in B:
        B.append(float(p))
    B = np.array(sorted(B))

    rng = np.random.default_rng(0)
    for _ in range(n_rollouts):
        b = float(p)
        for d in range(depth):
            # Expand via passive outcomes
            pt = predict_belief(b, a=0, P01=P01, P11=P11)
            py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
            # add both successors (with tiny jitter to avoid duplicates at edges)
            for cand in (p0_next, p1_next):
                cand = float(np.clip(cand, 0.0, 1.0))
                if np.min(np.abs(B - cand)) > 1e-6:
                    B = np.sort(np.append(B, cand + rng.uniform(-jitter, jitter)))
            # Randomly switch to active sometimes to include 0/1 collapses
            if rng.random() < 0.3:
                pt_act = predict_belief(b, a=1, P01=P01, P11=P11)
                # active leads to 0 with 1-pt_act, to 1 with pt_act (already in B)
                pass
            # move forward with a sampled passive observation
            b = p1_next if rng.random() < py1 else p0_next

    B = np.clip(B, 0.0, 1.0)
    B = np.unique(np.round(B, 8))  # dedup nicely
    S = len(B)

    # ---- 2) Build local transition models on this point set ----
    # Active: next belief is 0 or 1 with probs 1-pt, pt
    P_active = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=1, P01=P01, P11=P11)
        # P_active[i, _nn_index(0.0, B)] += (1.0 - pt)
        # P_active[i, _nn_index(1.0, B)] += pt
        for j, wj in split_to_grid(0.0, B):
            P_active[i, j] += (1.0 - pt) * wj
        for j, wj in split_to_grid(1.0, B):
            P_active[i, j] += pt * wj


    # Passive: two-posteriors kernel, mapped to nearest neighbors in B
    P_passive = np.zeros((S, S))
    for i, bi in enumerate(B):
        pt = predict_belief(bi, a=0, P01=P01, P11=P11)
        py0, p0_next, py1, p1_next = passive_posterior_terms(pt, th0, th1)
        # P_passive[i, _nn_index(np.clip(p0_next, 0, 1), B)] += py0
        # P_passive[i, _nn_index(np.clip(p1_next, 0, 1), B)] += py1
        for j, wj in split_to_grid(p0_next, B):
            P_passive[i, j] += py0 * wj
        for j, wj in split_to_grid(p1_next, B):
            P_passive[i, j] += py1 * wj
    # (rows should be stochastic; tiny drift is ok)

    # ---- 3) Rewards on the set ----
    r0 = B.copy()                 # r(p,0) = p
    r1 = B.copy() - float(lamb_val)  # r(p,1) = p - λ

    # ---- 4) Value iteration on the small set ----
    V  = np.zeros(S)
    for _ in range(max_iter):
        V_old = V.copy()
        Q0 = r0 + discount * (P_passive @ V_old)
        Q1 = r1 + discount * (P_active  @ V_old)
        V  = np.maximum(Q0, Q1)
        if np.max(np.abs(V - V_old)) < vi_tol:
            break

    # ---- 5) Return Q at the current p (nearest neighbor on B) ----
    ip = _nn_index(float(p), B)
    Q0_p, Q1_p = float(Q0[ip]), float(Q1[ip])
    a_star = int(Q1_p >= Q0_p)
    return Q0_p, Q1_p, a_star, B

    
def mean_pbvi_whittle_index(p_star, gamma, th0, th1, P01, P11, rng,
                            lam_lo=-1.0, lam_hi=1.0, tol=1e-5, max_iter=50):
    """Fast Whittle index with caching and warm-started VI/bisection."""
    global _LAST_LAMBDA, _V_CACHE

    # 1) local model (cached)
    B, P0, P1 = get_or_build_local_model(
        p_star, th0, th1, P01, P11, rng, n_rollouts=8, depth=3, active_reveals_H=True
    )
    key_V = (tuple(np.round(B, 10)), _q(gamma, 1e-6))
    V0 = _V_CACHE.get(key_V)

    # 2) bracket around last lambda if available
    if _LAST_LAMBDA is not None:
        mid = float(_LAST_LAMBDA)
        lo, hi = max(lam_lo, mid - 0.2), min(lam_hi, mid + 0.2)
        if lo >= hi:
            lo, hi = lam_lo, lam_hi
    else:
        lo, hi = lam_lo, lam_hi

    # 3) bisection with warm-start VI
    for _ in range(max_iter):
        lam = 0.5 * (lo + hi)
        V, Q0, Q1 = value_iter_with_warm_start(B, P0, P1, lam, gamma, V0)
        _V_CACHE[key_V] = V
        ip = int(np.argmin(np.abs(np.asarray(B) - float(p_star))))
        if Q1[ip] > Q0[ip]:
            lo = lam
        else:
            hi = lam
        if hi - lo < tol:
            break
        V0 = V

    w = 0.5 * (lo + hi)
    _LAST_LAMBDA = w
    return w
