"""Deep BSDE for the 100-dimensional Allen-Cahn equation.

    u_t + Laplacian(u) + u - u^3 = 0,   u(T, x) = 1 / (2 + 0.4 |x|^2)

with T = 0.3, solved for u(0, 0). Benchmark from Han, Jentzen & E (2018),
reference value u(0, 0) ~ 0.0528.

Run `python allen_cahn.py` to train, or `python allen_cahn.py --load`
to evaluate a saved checkpoint.
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

DIM = 100
NUM_STEPS = 20
T = 0.3
BATCH_SIZE = 64
NUM_EPOCHS = 3000
LR = 5e-4
REFERENCE_Y0 = 0.0528
CHECKPOINT = Path("allen_cahn_checkpoint.pt")
FIGURES = Path("figures")
DT = T / NUM_STEPS


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AllenCahnZNetwork(nn.Module):
    """Maps (t, X_t) in R^(d+1) to Z_t in R^d."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        # The true Z is close to zero here, so start the output there instead of
        # letting random initial weights inject noise through Z . dW.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def simulate_forward_paths(batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate X_t = sqrt(2) W_t from X_0 = 0, the diffusion behind the Laplacian."""
    dW = torch.randn(batch_size, NUM_STEPS, DIM, device=device) * math.sqrt(DT)
    X = torch.cumsum(math.sqrt(2.0) * dW, dim=1)
    X = torch.cat([torch.zeros(batch_size, 1, DIM, device=device), X], dim=1)
    return X, dW


def terminal_condition(X_T: torch.Tensor) -> torch.Tensor:
    return 1.0 / (2.0 + 0.4 * (X_T**2).sum(dim=-1))


def allen_cahn_step(Y, Z, dW_n) -> torch.Tensor:
    """One Euler step of dY = -f(Y) dt + Z . dW with driver f(y) = y - y^3."""
    return Y - (Y - Y**3) * DT + (Z * dW_n).sum(dim=-1)


def rollout(z_net, Y0, X, dW) -> torch.Tensor:
    batch_size = X.shape[0]
    Y = Y0.repeat(batch_size)
    for n in range(NUM_STEPS):
        t_col = torch.full((batch_size, 1), n * DT, device=X.device)
        Z = z_net(torch.cat([t_col, X[:, n, :]], dim=1))
        Y = allen_cahn_step(Y, Z, dW[:, n, :])
    return Y


def train(z_net, Y0, device) -> dict:
    optimizer = optim.Adam(list(z_net.parameters()) + [Y0], lr=LR)
    history = {"loss": [], "y0": []}

    for epoch in range(NUM_EPOCHS):
        X, dW = simulate_forward_paths(BATCH_SIZE, device)
        loss = ((rollout(z_net, Y0, X, dW) - terminal_condition(X[:, -1, :])) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["y0"].append(Y0.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | loss {loss.item():.6f} | Y0 {Y0.item():.6f}")

    torch.save(
        {
            "z_network": {k: v.detach().cpu() for k, v in z_net.state_dict().items()},
            "Y0": Y0.detach().cpu(),
            "history": history,
        },
        CHECKPOINT,
    )
    print(f"Saved {CHECKPOINT}")
    return history


def evaluate(z_net, Y0, device) -> None:
    X, dW = simulate_forward_paths(10_000, device)
    with torch.no_grad():
        error = rollout(z_net, Y0, X, dW) - terminal_condition(X[:, -1, :])

    print("\nFresh-sample validation (10,000 paths)")
    print(f"Learned Y0:      {Y0.item():.5f}")
    print(f"Reference Y0:    {REFERENCE_Y0}")
    print(f"Relative error:  {abs(Y0.item() - REFERENCE_Y0) / REFERENCE_Y0:.2%}")
    print(f"Terminal RMSE:   {error.pow(2).mean().sqrt().item():.5f}")


def plot_history(history: dict) -> None:
    FIGURES.mkdir(exist_ok=True)
    epochs = range(len(history["loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["y0"], label="Learned $Y_0$")
    ax.axhline(REFERENCE_Y0, linestyle="--", color="C1", label=f"Reference $Y_0 \\approx$ {REFERENCE_Y0}")
    ax.set(xlabel="Epoch", ylabel="$Y_0$", title="Allen-Cahn 100D: $Y_0$ convergence")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "allen_cahn_y0_convergence.png", dpi=200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs, history["loss"])
    ax.set(xlabel="Epoch", ylabel="Loss (log scale)", title="Allen-Cahn 100D: training loss")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "allen_cahn_training_loss.png", dpi=200)
    print(f"Saved Allen-Cahn figures to {FIGURES}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true", help="evaluate saved checkpoint")
    args = parser.parse_args()

    device = get_device()
    torch.manual_seed(42)

    z_net = AllenCahnZNetwork(DIM).to(device)
    Y0 = nn.Parameter(torch.tensor(0.1, device=device))

    if args.load:
        checkpoint = torch.load(CHECKPOINT, map_location=device)
        z_net.load_state_dict(checkpoint["z_network"])
        with torch.no_grad():
            Y0.copy_(checkpoint["Y0"].to(device))
        history = checkpoint.get("history")
    else:
        history = train(z_net, Y0, device)

    z_net.eval()
    evaluate(z_net, Y0, device)
    if history:
        tail = torch.tensor(history["y0"][-500:])
        print(f"Y0, last 500 epochs: {tail.mean():.5f} ± {tail.std():.5f}")
        plot_history(history)
    else:
        print("Checkpoint has no training history; retrain to regenerate the plots.")


if __name__ == "__main__":
    main()
