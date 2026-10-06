"""Terminal error of the BSDE recursion when Z is the exact analytic hedge.

With no learning error left, the mismatch Y_T - payoff is pure discrete-
hedging error, which should shrink like 1 / sqrt(N). This sets the floor the
trained network cannot beat at a given number of time steps.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from bs_utils import black_scholes_call_price, call_payoff, rollout_exact_z, simulate_stock_paths

S0, STRIKE, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
STEP_COUNTS = [10, 20, 50, 100, 200, 400]
NUM_PATHS = 20_000


def main() -> None:
    torch.manual_seed(0)
    Y0 = black_scholes_call_price(S0, STRIKE, R, SIGMA, T)
    rmses = []

    for N in STEP_COUNTS:
        paths, dW = simulate_stock_paths(NUM_PATHS, N, T, S0, R, SIGMA)
        error = rollout_exact_z(paths, dW, STRIKE, R, SIGMA, T, Y0) - call_payoff(paths[:, -1], STRIKE)
        rmse = error.pow(2).mean().sqrt().item()
        rmses.append(rmse)
        print(f"N = {N:4d} | terminal RMSE {rmse:.4f}")

    reference = [rmses[0] * (STEP_COUNTS[0] / N) ** 0.5 for N in STEP_COUNTS]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(STEP_COUNTS, rmses, "o-", label="Exact-Z BSDE")
    ax.loglog(STEP_COUNTS, reference, "--", label=r"$\propto N^{-1/2}$")
    ax.set(xlabel="Time steps N", ylabel="Terminal RMSE", title="Discrete hedging error")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/discretization_error.png", dpi=200)
    print("Saved figures/discretization_error.png")


if __name__ == "__main__":
    main()
