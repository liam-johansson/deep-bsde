"""Deep BSDE for a 1D European call, validated against Black-Scholes.

Run `python black_scholes.py` to train, or `python black_scholes.py --load`
to evaluate a saved checkpoint.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from bs_utils import (
    ZNetwork,
    black_scholes_call_delta,
    black_scholes_call_price,
    bsde_step,
    call_payoff,
    get_device,
    simulate_stock_paths,
)

S0, STRIKE, R, SIGMA, T = 100.0, 100.0, 0.05, 0.20, 1.0
NUM_STEPS = 50
NUM_PATHS = 10_000
NUM_EPOCHS = 1500
LR = 0.01
CHECKPOINT = Path("deep_bsde_checkpoint.pt")
FIGURES = Path("figures")


def network_input(t: float, S: torch.Tensor) -> torch.Tensor:
    """Stack (t, (S - K) / K) so the inputs are O(1)."""
    return torch.stack([torch.full_like(S, t), (S - STRIKE) / STRIKE], dim=1)


def rollout(z_net, Y0, paths, dW) -> torch.Tensor:
    """Run the BSDE forward along each path using the network's Z."""
    dt = T / NUM_STEPS
    Y = Y0.repeat(paths.shape[0])
    for n in range(NUM_STEPS):
        Z = z_net(network_input(n * dt, paths[:, n])).squeeze(1)
        Y = bsde_step(Y, Z, dW[:, n], R, dt)
    return Y


def train(z_net, Y0, device) -> None:
    optimizer = optim.Adam(list(z_net.parameters()) + [Y0], lr=LR)
    for epoch in range(NUM_EPOCHS):
        paths, dW = simulate_stock_paths(NUM_PATHS, NUM_STEPS, T, S0, R, SIGMA, device)
        loss = ((rollout(z_net, Y0, paths, dW) - call_payoff(paths[:, -1], STRIKE)) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | loss {loss.item():.4f} | Y0 {Y0.item():.4f}")

    torch.save(
        {
            "z_network": {k: v.detach().cpu() for k, v in z_net.state_dict().items()},
            "Y0": Y0.detach().cpu(),
        },
        CHECKPOINT,
    )
    print(f"Saved {CHECKPOINT}")


def evaluate(z_net, Y0, device) -> None:
    """Out-of-sample terminal error on fresh paths, and Y0 vs the closed form."""
    paths, dW = simulate_stock_paths(50_000, NUM_STEPS, T, S0, R, SIGMA, device)
    with torch.no_grad():
        error = rollout(z_net, Y0, paths, dW) - call_payoff(paths[:, -1], STRIKE)

    reference = black_scholes_call_price(S0, STRIKE, R, SIGMA, T)
    print("\nFresh-sample validation (50,000 paths)")
    print(f"Learned Y0:        {Y0.item():.4f}")
    print(f"Black-Scholes Y0:  {reference:.4f}")
    print(f"Absolute error:    {abs(Y0.item() - reference):.4f}")
    print(f"Terminal RMSE:     {error.pow(2).mean().sqrt().item():.4f}")


def plot_z(z_net, device) -> None:
    """Learned Z(t, S) against the analytic sigma * S * delta."""
    S = torch.linspace(70.0, 130.0, 200)
    fig, ax = plt.subplots(figsize=(8, 5))

    for t, color in [(0.25, "C0"), (0.5, "C1"), (0.75, "C2")]:
        with torch.no_grad():
            learned = z_net(network_input(t, S).to(device)).squeeze(1).cpu()
        exact = SIGMA * S * black_scholes_call_delta(S, STRIKE, R, SIGMA, T - t)
        ax.plot(S, exact, color=color, label=f"Analytic, t = {t}")
        ax.plot(S, learned, color=color, linestyle="--", label=f"Network, t = {t}")

    ax.set(xlabel="Stock price S", ylabel="Z(t, S)", title="Learned vs analytic Z")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "black_scholes_z_validation.png", dpi=200)
    print(f"Saved {FIGURES / 'black_scholes_z_validation.png'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="evaluate saved checkpoint")
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(42)

    z_net = ZNetwork().to(device)
    Y0 = nn.Parameter(torch.tensor(5.0, device=device))

    if args.load:
        checkpoint = torch.load(CHECKPOINT, map_location=device)
        z_net.load_state_dict(checkpoint["z_network"])
        with torch.no_grad():
            Y0.copy_(checkpoint["Y0"].to(device))
    else:
        train(z_net, Y0, device)

    z_net.eval()
    evaluate(z_net, Y0, device)
    plot_z(z_net, device)


if __name__ == "__main__":
    main()
