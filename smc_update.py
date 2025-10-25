import numpy as np
from scipy.special import logsumexp
from utils import *

def normalize_weights(w):
    eps=1e-16
    log_w = np.log(w + eps)  # convert to log-space
    log_w -= logsumexp(log_w)
    w = np.exp(log_w)
    return w 

def ESS(w):
    w = normalize_weights(w)
    return 1.0 / np.sum(w**2)

def systematic_resample(theta0, theta1, w, M_keep, rng):
    """Return resampled smaller (M_keep) sets of particles with equal weights
    """
    w = normalize_weights(w)
    cdf = np.cumsum(w)
    u0 = rng.random() / M_keep
    us = u0 + np.arange(M_keep) / M_keep
    idx = np.searchsorted(cdf, us, side="right")
    return theta0[idx].copy(), theta1[idx].copy(), np.ones(M_keep) / M_keep, idx


def calculate_weighted_sd(theta0, theta1, w):
    mu_w0 = np.sum(w * theta0)
    var_theta0 = np.sum(w * (theta0 - mu_w0) ** 2)
    sd_theta0 = np.sqrt(var_theta0)
    
    mu_w1 = np.sum(w * theta1)
    var_theta1 = np.sum(w * (theta1 - mu_w1) ** 2)
    sd_theta1 = np.sqrt(var_theta1)
    
    return sd_theta0, sd_theta1
 

def smc_update_single(
    p,
    action,
    y,
    P01,
    P11,
    theta0,
    theta1,
    w,
    rng,
    H=None,
    ess_ratio=0.5,
    lw_a=None,
    M_keep=40,
    eps=1e-16,
):
    """
    One-step SMC update for (theta0, theta1) given a single arm, one time step,
    the taken action, and the observed message y.

    Parameters
    - p_cur: scalar prior belief P(s_t=1) before observing y (after transition)
             If you have belief at t-1, set p_cur to the predicted belief using
             the provided transition and action.
    - action: int in {0,1,...}, action applied on this arm (affects transition only)
    - y: int in {0,1}, observed message at time t
    - trans_mat: array with shape [2, 2, 2], P(s_t | s_{t-1}, action)
    - theta0, theta1: arrays of particles in (0,1), length M
    - w: normalized particle weights, length M 
    - rng: np.random.Generator for resampling and jitter
    H: observed health status 
    - ess_ratio: resample when ESS < ess_ratio * M
    - lw_a: Liu–West shrinkage coefficient in (0,1); if None, no jitter
    - M_keep: number of particles to keep for downstream value computation

    Returns
    - p_mean (scalar, belief mean after update)
    - theta0, theta1, w
    """
    M = theta0.shape[0]
    assert theta1.shape[0] == M and w.shape[0] == M

    a = int(action)
    
    # Predict belief with the given action (from t-1 to t)
    p_pred = predict_belief(p, a, P01, P11)

    # Update weights (log-space stable)
    log_w = np.log(w + eps)
    if H is not None:
         if int(H) == 0:
            lik = theta0 if int(y) == 1 else (1.0 - theta0)
         else:
            lik = theta1 if int(y) == 1 else (1.0 - theta1)
         log_w += np.log(lik + eps)
         log_w -= logsumexp(log_w)
         w = np.exp(log_w)
         p_mean = float(H)
         # ESS resampling/jitter (do NOT try to index p_mean)
         if ESS(w) < ess_ratio * M:
            idx = rng.choice(M, size=M, replace=True, p=w)
            theta0, theta1 = theta0[idx], theta1[idx]
            w = np.full(M, 1.0 / M)

            if lw_a is not None:
                for th in (theta0, theta1):
                    m = th.mean()
                    var = th.var()
                    b2 = max(0.0, (1.0 - lw_a**2) * var)
                    th += (lw_a - 1.0) * (th - m) + rng.normal(0.0, np.sqrt(b2 + eps), size=M)
                    np.clip(th, 1e-4, 1.0 - 1e-4, out=th)

         return theta0, theta1, w, p_mean 
         
    else:
        # ----- Soft/mixture likelihood and per-particle posterior -----
        if int(y) == 1:
            num   = p_pred * theta1
            lik = (1.0 - p_pred) * theta0 + num
        else:
            num   = p_pred * (1.0 - theta1)
            lik = (1.0 - p_pred) * (1.0 - theta0) + num

        # Weight update
        log_w += np.log(lik + eps)
        log_w -= logsumexp(log_w)
        w = np.exp(log_w)

        # Per-particle posterior p(H_t=1 | y, theta)
        p_post_particles = num / (lik + eps)
        # Posterior mean belief
        p_mean = float(np.sum(w * p_post_particles))

        # ESS resampling/jitter (resample particles; recompute uniform w)
        if ESS(w) < ess_ratio * M:
            idx = rng.choice(M, size=M, replace=True, p=w)
            theta0, theta1, p_post_particles = theta0[idx], theta1[idx], p_post_particles[idx]
            w = np.full(M, 1.0 / M)

            if lw_a is not None:
                for th in (theta0, theta1):
                    m = th.mean()
                    var = th.var()
                    b2 = max(0.0, (1.0 - lw_a**2) * var)
                    th += (lw_a - 1.0) * (th - m) + rng.normal(0.0, np.sqrt(b2 + eps), size=M)
                    np.clip(th, 1e-4, 1.0 - 1.0e-4, out=th)
        
        return theta0, theta1, w, p_mean 
