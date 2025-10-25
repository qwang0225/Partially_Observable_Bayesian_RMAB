import numpy as np


def compute_priority_scores(p_belief, th0_mean, th1_mean, P01_all, P11_all,
                            gamma, rng=None, lam_explore=0.0):
    """
    Priority index using posterior mean theta per arm (no per-particle eval),
    with a closed-form value function V(b) = b / (1 - gamma).

    For each arm i with belief b and mean thetas (th0, th1):
      - V(b) = b / (1 - gamma)
      - EV_active = b * V(P11_act) + (1 - b) * V(P01_act)
      - Passive: tilde = b*P11_pass + (1-b)*P01_pass,
                 pi1 = tilde*th1 + (1-tilde)*th0,
                 p1  = (tilde*th1)/pi1,
                 p0  = (tilde*(1-th1))/(1-pi1),
                 EV_pass = pi1*V(p1) + (1-pi1)*V(p0)
      - Priority score = gamma*(EV_active - EV_pass) + lam_explore*sqrt(pi1*(1-pi1))
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(p_belief)
    scores = np.zeros(N, dtype=float)
    inv_one_minus_gamma = 1.0 / (1.0 - float(gamma) + 1e-16)

    for i in range(N):
        b = float(p_belief[i])
        th0 = float(th0_mean[i])
        th1 = float(th1_mean[i])
        P01 = P01_all[i]
        P11 = P11_all[i]

        # Closed-form value function
        def V(x):
            return float(np.clip(x, 0.0, 1.0)) * inv_one_minus_gamma

        # Active continuation
        V_act = b * V(P11[1]) + (1.0 - b) * V(P01[1])

        # Passive continuation via message mixture
        tilde = b * P11[0] + (1.0 - b) * P01[0]
        tilde = float(np.clip(tilde, 0.0, 1.0))
        pi1 = tilde * th1 + (1.0 - tilde) * th0
        pi1 = float(np.clip(pi1, 1e-12, 1.0 - 1e-12))
        p1_post = (tilde * th1) / pi1
        p0_post = (tilde * (1.0 - th1)) / (1.0 - pi1)
        p1_post = float(np.clip(p1_post, 0.0, 1.0))
        p0_post = float(np.clip(p0_post, 0.0, 1.0))
        V_pas = pi1 * V(p1_post) + (1.0 - pi1) * V(p0_post)

        delta = float(gamma) * (V_act - V_pas)
        bonus = float(lam_explore) * float(np.sqrt(pi1 * (1.0 - pi1))) if lam_explore > 0 else 0.0
        scores[i] = delta + bonus

    return scores


def select_priority_actions(p_belief, budget, th0_mean, th1_mean, P01_all, P11_all,
                            gamma, lam_explore=0.0, rng=None):
    scores = compute_priority_scores(
        p_belief=p_belief,
        th0_mean=th0_mean,
        th1_mean=th1_mean,
        P01_all=P01_all,
        P11_all=P11_all,
        gamma=gamma,
        rng=rng,
        lam_explore=lam_explore,
    )
    N = len(p_belief)
    actions = np.zeros(N, dtype=int)
    k = int(min(budget, N))
    if k > 0:
        idx = np.argpartition(-scores, k - 1)[:k]
        actions[idx] = 1
    return actions
