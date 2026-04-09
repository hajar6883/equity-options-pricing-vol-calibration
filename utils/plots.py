import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_iv_comparison(m_grid, maturities, Z_mkt, Z_model, title="IV: Market vs Model"):
    """
    Side-by-side IV surface plots: market on the left, model on the right.
    Also prints per-maturity RMSE to stdout.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={"projection": "3d"})
    M, T = np.meshgrid(m_grid, maturities)

    vmin = np.nanmin([Z_mkt, Z_model])
    vmax = np.nanmax([Z_mkt, Z_model])

    for ax, Z, label in zip(axes, [Z_mkt, Z_model], ["Market", "Model"]):
        surf = ax.plot_surface(M, T, Z, cmap=cm.viridis, vmin=vmin, vmax=vmax, alpha=0.85)
        ax.set_xlabel("Moneyness m = K/F")
        ax.set_ylabel("T (years)")
        ax.set_zlabel("IV")
        ax.set_title(label)

    fig.colorbar(surf, ax=axes, shrink=0.5, label="IV")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_iv_slices(m_grid, maturities, Z_mkt, Z_model, n_slices=6):
    """
    Overlay market and model IV smiles for a selection of maturities.
    More readable than a 3-D surface for spotting where the fit breaks.
    """
    indices = np.linspace(0, len(maturities) - 1, min(n_slices, len(maturities)), dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax_idx, i in enumerate(indices):
        ax = axes[ax_idx]
        T = maturities[i]

        mkt = Z_mkt[i]
        mdl = Z_model[i]
        mask = np.isfinite(mkt) & np.isfinite(mdl)

        ax.plot(m_grid[mask], mkt[mask], "o-", label="Market", markersize=3)
        ax.plot(m_grid[mask], mdl[mask], "s--", label="Model", markersize=3)
        ax.axvline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"T = {T:.2f}y")
        ax.set_xlabel("m = K/F")
        ax.set_ylabel("IV")
        ax.legend(fontsize=8)

    for ax in axes[len(indices):]:
        ax.set_visible(False)

    plt.suptitle("IV smile slices: Market vs Model")
    plt.tight_layout()
    plt.show()


def plot_rmse_by_maturity(maturities, rmse_arr, title="IV RMSE by Maturity"):
    """
    Bar chart of per-maturity RMSE.
    The first thing a reviewer looks at to judge calibration quality.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(maturities))
    bars = ax.bar(x, rmse_arr * 100, color="steelblue", edgecolor="white")

    ax.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{T:.2f}y" for T in maturities], rotation=45)
    ax.set_ylabel("IV RMSE (%%)")
    ax.set_xlabel("Maturity")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
