"""Black-Scholes building blocks for the Deep BSDE experiments."""

import math

import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def call_payoff(terminal_prices: torch.Tensor, strike: float) -> torch.Tensor:
    """European call payoff max(S_T - K, 0)."""
    return torch.clamp(terminal_prices - strike, min=0)


def simulate_stock_paths(
    num_paths: int,
    num_steps: int,
    terminal_time: float,
    S0: float,
    r: float,
    sigma: float,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate risk-neutral GBM paths with the exact log-normal step.

    Returns the paths, shape (num_paths, num_steps + 1), and the Brownian
    increments that drove them, shape (num_paths, num_steps). The same
    increments must be reused in the BSDE recursion.
    """
    device = device or torch.device("cpu")
    dt = terminal_time / num_steps

    dW = torch.randn(num_paths, num_steps, device=device) * math.sqrt(dt)

    paths = torch.empty(num_paths, num_steps + 1, device=device)
    paths[:, 0] = S0
    for n in range(num_steps):
        paths[:, n + 1] = paths[:, n] * torch.exp(
            (r - 0.5 * sigma**2) * dt + sigma * dW[:, n]
        )

    return paths, dW


def black_scholes_call_price(
    S0: float, strike: float, r: float, sigma: float, terminal_time: float
) -> float:
    """Closed-form Black-Scholes price of a European call."""
    sqrt_T = math.sqrt(terminal_time)
    d1 = (math.log(S0 / strike) + (r + 0.5 * sigma**2) * terminal_time) / (
        sigma * sqrt_T
    )
    d2 = d1 - sigma * sqrt_T

    def N(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    return S0 * N(d1) - strike * math.exp(-r * terminal_time) * N(d2)


def black_scholes_call_delta(
    S: torch.Tensor, strike: float, r: float, sigma: float, time_to_maturity: float
) -> torch.Tensor:
    """Black-Scholes call delta N(d1), evaluated elementwise on a tensor of spots."""
    d1 = (torch.log(S / strike) + (r + 0.5 * sigma**2) * time_to_maturity) / (
        sigma * math.sqrt(time_to_maturity)
    )
    # erf instead of torch.distributions keeps everything on S.device.
    return 0.5 * (1.0 + torch.erf(d1 / math.sqrt(2.0)))


def bsde_step(
    Y: torch.Tensor, Z: torch.Tensor, dW: torch.Tensor, r: float, dt: float
) -> torch.Tensor:
    """One Euler step of dY = rY dt + Z dW (self-financing replicating portfolio)."""
    return Y + r * Y * dt + Z * dW


def rollout_exact_z(
    paths: torch.Tensor,
    dW: torch.Tensor,
    strike: float,
    r: float,
    sigma: float,
    terminal_time: float,
    Y0: float,
) -> torch.Tensor:
    """Run the BSDE forward using the analytic Z = sigma * S * delta.

    The terminal mismatch Y_T - payoff then measures pure time-discretization
    (discrete hedging) error, with no learning error involved.
    """
    num_steps = dW.shape[1]
    dt = terminal_time / num_steps
    Y = torch.full((paths.shape[0],), Y0, device=paths.device, dtype=paths.dtype)

    for n in range(num_steps):
        S_n = paths[:, n]
        delta = black_scholes_call_delta(S_n, strike, r, sigma, terminal_time - n * dt)
        Y = bsde_step(Y, sigma * S_n * delta, dW[:, n], r, dt)

    return Y


class ZNetwork(nn.Module):
    """Maps (t, normalized S) to Z(t, S) for the 1D Black-Scholes problem."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
