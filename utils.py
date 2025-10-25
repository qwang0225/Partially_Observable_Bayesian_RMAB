import numpy as np
import matplotlib.pyplot as plt


def predict_belief(p_val, a, P01, P11):
        """Predictive health belief before observing at this time step."""
        return (1.0 - p_val) * P01[a] + p_val * P11[a]

def split_to_grid(x, grid):
        """Return list[(index, weight)] to distribute mass linearly onto grid."""
        if x <= grid[0]:
                return [(0, 1.0)]
        if x >= grid[-1]:
                return [(len(grid) - 1, 1.0)]
        j = np.searchsorted(grid, x)
        j0, j1 = j - 1, j
        w = (x - grid[j0]) / (grid[j1] - grid[j0])
        return [(j0, 1.0 - w), (j1, w)]

def passive_posterior_terms(pt, th0, th1):
    # Message likelihoods and posteriors after Y in {0,1}
    py1 = pt*th1 + (1.0 - pt)*th0
    py0 = 1.0 - py1

    # p'(H=1 | Y=1)
    if py1 > 0:
        p1 = (pt*th1) / py1
    else:
        p1 = 1.0 if pt > 0 else 0.0

    # p'(H=1 | Y=0)
    d0 = pt*(1.0 - th1) + (1.0 - pt)*(1.0 - th0)
    if d0 > 0:
        p0 = (pt*(1.0 - th1)) / d0
    else:
        p0 = 1.0 if pt > 0 else 0.0

    return py0, p0, py1, p1

def beta_paras(mean, k):
    """
    mean: alpha / (alpha + beta)
    k: concentration = alpha + beta 
    return alpha, beta 
    """
    return mean *k, (1-mean)*k

def plot_theta_tracking_pvbi(theta0_true_over_eps, theta1_true_over_eps,
                             theta0_est_over_eps, theta1_est_over_eps,
                             arms_to_plot=5, title=None, out_path=None,
                             sample_random=True, sample_seed=None):
    """Plot true theta0/theta1 over episodes and estimated trajectories for selected arms.

    - theta0_true_over_eps: array (episodes, num_arms)
    - theta1_true_over_eps: array (episodes, num_arms)
    - theta0_est_over_eps:  array (episodes, num_arms)
    - theta1_est_over_eps:  array (episodes, num_arms)
    """
    if theta0_est_over_eps is None or theta1_est_over_eps is None:
        return

    episodes = theta0_est_over_eps.shape[0]
    num_arms = theta0_est_over_eps.shape[1]

    k = int(min(max(1, arms_to_plot), num_arms))
    if sample_random and num_arms > 0:
        rng = np.random.default_rng(sample_seed)
        sel_arms = rng.choice(num_arms, size=k, replace=False)
    else:
        sel_arms = np.arange(k)

    x = np.arange(1, episodes + 1)

    fig, (ax0, ax1, axm) = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # Theta0 subplot
    for j in sel_arms:
        line0, = ax0.plot(x, theta0_est_over_eps[:, j], label=f"arm {j}")
        ax0.plot(x, theta0_true_over_eps[:, j], linestyle="dashed", color=line0.get_color())
    ax0.set_title("Theta0: est (solid) vs true (dashed)")
    ax0.set_xlabel("Episode")
    ax0.set_ylabel("theta0")

    # Theta1 subplot
    for j in sel_arms:
        line1, = ax1.plot(x, theta1_est_over_eps[:, j], label=f"arm {j}")
        ax1.plot(x, theta1_true_over_eps[:, j], linestyle="dashed", color=line1.get_color())
    ax1.set_title("Theta1: est (solid) vs true (dashed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("theta1")

    # Aggregate MAE over all arms per episode
    mae0 = np.mean(np.abs(theta0_est_over_eps - theta0_true_over_eps), axis=1)
    mae1 = np.mean(np.abs(theta1_est_over_eps - theta1_true_over_eps), axis=1)
    axm.plot(x, mae0, label="MAE theta0")
    axm.plot(x, mae1, label="MAE theta1")
    axm.set_title("Aggregate MAE over episodes")
    axm.set_xlabel("Episode")
    axm.set_ylabel("MAE")
    axm.legend()

    # One legend for arm traces (limit clutter)
    handles, labels = ax1.get_legend_handles_labels()
    if k <= 10:
        fig.legend(handles, labels, loc="upper center", ncol=min(k, 5))

    if title:
        fig.suptitle(title)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")


def plot_results(results, policy_list):
    """Plot comparison results."""
    labels = [f"N={N},B={B}" for N, B in results.keys()]
    x = np.arange(len(labels))
    # Use only policies requested and present in results; use lowercase labels consistently
    policy_order = [(p, p) for p in policy_list if all(p in results[k] for k in results.keys())]
    m = max(1, len(policy_order))
    width = 0.8 / m

    means_by_pol = { key: [results[k][key]["mean"] for k in results.keys()] for key, _ in policy_order }
    stds_by_pol  = { key: [results[k][key]["std"]  for k in results.keys()] for key, _ in policy_order }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot means
    offsets = np.linspace(-(m-1)/2, (m-1)/2, num=m) * width
    bars_all = []
    for (offset, (pol_key, pol_label)) in zip(offsets, policy_order):
        bars = ax1.bar(x + offset, means_by_pol[pol_key], width, label=pol_label, alpha=0.8)
        bars_all.append(bars)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel("Average Discounted Return")
    ax1.set_title("Policy Comparison: Average Returns")
    ax1.legend()
    
    # Add value labels on bars
    for bars in bars_all:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom')

    # Plot standard deviations
    for (offset, (pol_key, pol_label)) in zip(offsets, policy_order):
        ax2.bar(x + offset, stds_by_pol[pol_key], width, label=pol_label, alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel("Standard Deviation of Returns")
    ax2.set_title("Policy Comparison: Return Variability")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("policy_comparison.png", dpi=200, bbox_inches="tight")


def plot_runtime(results, policy_list):
    labels = [f"N={N},B={B}" for N, B in results.keys()]
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(policy_list))

    def safe_list(pol):
        return [results[key][pol].get("runtime_sec", np.nan) for key in results.keys()]

    order = [(p, p) for p in policy_list if all(p in results[k] for k in results.keys())]

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    offsets = np.linspace(-(len(order)-1)/2, (len(order)-1)/2, num=len(order)) * width
    for (offset, (pol_key, pol_label)) in zip(offsets, order):
        ax.bar(x + offset, safe_list(pol_key), width, label=pol_label, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Runtime (s)")
    ax.set_title("Runtime Comparison per Configuration")
    ax.legend()
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width()/2., h, f'{h:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("runtime_comparison.png", dpi=200, bbox_inches="tight")


def print_results_summary(results, policy_list):
    """Print a concise, lower-case summary for the selected policies."""
    print("\n" + "=" * 50)
    print("detailed results summary")
    print("=" * 50)

    # Match filtering used in plots
    policy_order = [(p, p) for p in policy_list if all(p in results[k] for k in results.keys())]

    for key in results.keys():
        N, B = key
        print(f"\nconfiguration: N={N}, B={B}")
        random_mean = results[key]["random"]["mean"] if "random" in results[key] else None
        for pol, _ in policy_order:
            mean_val = results[key][pol]["mean"]
            std_val = results[key][pol]["std"]
            runtime = results[key][pol].get("runtime_sec", float("nan"))
            if random_mean is not None and random_mean != 0:
                improvement = (mean_val - random_mean) / random_mean * 100
                extra = f"(vs random: {improvement:+.1f}%)"
            else:
                extra = ""
            print(f"  {pol:12}: {mean_val:.3f} +/- {std_val:.3f}  (rt: {runtime:.2f}s) {extra}")
