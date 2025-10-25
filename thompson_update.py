from utils import *


def ts_soft_update_single(p_cur, y, action, P01, P11,
                          a0, b0, a1, b1, rng=None):
    """
    One-step Rao-Blackwellized (soft-count) Thompson update for ONE arm
    with predict-then-observe ordering.

    Args:
      p_cur : float
          Previous-step belief p_{t-1} = P(H_{t-1} = 1).
      y : int in {0,1}
          Observed message at time t.
      action : int in {0,1}
          Action taken at time t (indexes P01/P11).
      P01 : array-like of shape (2,)
          P(H_{t+1}=1 | H_t=0, a), indexed by action (0 or 1).
      P11 : array-like of shape (2,)
          P(H_{t+1}=1 | H_t=1, a), indexed by action (0 or 1).
      a0,b0,a1,b1 : floats
          Beta params for theta_0 (unhealthy) and theta_1 (healthy).
      rng : numpy Generator (optional)
          Random generator used to sample theta parameters.

    Returns:
      a0_new, b0_new, a1_new, b1_new, p_next, r_t
        Updated Beta params, next-step belief p_{t+1}, and posterior r_t=P(H_t=1|y).
    """
    # 1) Sample theta parameters (Thompson)
    if rng is None:
        import numpy as _np
        rng = _np.random.default_rng()
    th0 = rng.beta(a0, b0)
    th1 = rng.beta(a1, b1)

    # 2) Predict to time t (before observing y): p_pred = P(H_t=1)
    a = int(action)
    p_pred = predict_belief(p_cur, a, P01, P11)

    # 3) Filter: posterior over H_t given y and sampled thetas
    if int(y) == 1:
        num = p_pred * th1
        den = num + (1.0 - p_pred) * th0
    else:  # y == 0
        num = p_pred * (1.0 - th1)
        den = num + (1.0 - p_pred) * (1.0 - th0)
    p_post = 0.0 if den == 0.0 else num / den

    # 4) Soft-count Beta updates
    a0_new, b0_new, a1_new, b1_new = a0, b0, a1, b1
    if int(y) == 1:
        a1_new += p_post
        a0_new += (1.0 - p_post)
    else: 
        b1_new += p_post
        b0_new += (1.0 - p_post)
    
    return a0_new, b0_new, a1_new, b1_new, p_post

def ts_labeled_update_single(H, y, a0, b0, a1, b1):
    """
    update with hard label when action = 1, fully observed health status
    """
    if H == 1:
        a1 += (y == 1)
        b1 += (y == 0)
    else:
        a0 += (y == 1)
        b0 += (y == 0)
    return a0, b0, a1, b1
