"""Discover damping in a nonlinear pendulum from noisy angle measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def generate_data(seed, count, noise, duration, damping, wind):
    """Integrate theta'' + b theta' + (g/L) sin(theta) = 0."""
    rng = np.random.default_rng(seed)
    times = np.linspace(0, duration, count)
    dense_times = np.linspace(0, duration, 501)
    # Solve the nonlinear pendulum with an optional periodic wind torque.
    solution = solve_ivp(lambda t, y: (y[1], -damping*y[1] - 9.81*np.sin(y[0]) + wind*np.sin(1.5*t)),
                         (0, duration), (1.0, 0.0), t_eval=dense_times,
                         rtol=1e-10, atol=1e-10, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    # Sample sparse measurements separately from the dense evaluation curve.
    clean = solution.sol(times)[0]
    return times, clean + rng.normal(0, noise, count), dense_times, solution.y[0]


class PendulumPINN(nn.Module):
    def __init__(self, duration, width):
        super().__init__()
        self.duration = duration
        self.network = nn.Sequential(nn.Linear(1, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(),
                                     nn.Linear(width, 1))
        # Softplus keeps the learned physical damping nonnegative.
        self.raw_damping = nn.Parameter(torch.tensor(-1.5))

    @property
    def damping(self):
        return torch.nn.functional.softplus(self.raw_damping)

    def forward(self, t):
        return self.network(2 * t / self.duration - 1)


def residual(model, t, wind=0.0):
    # Differentiate the predicted angle twice with respect to time.
    t = t.detach().clone().requires_grad_(True)
    theta = model(t)
    velocity = torch.autograd.grad(theta, t, torch.ones_like(theta), create_graph=True)[0]
    acceleration = torch.autograd.grad(velocity, t, torch.ones_like(velocity), create_graph=True)[0]
    return acceleration + model.damping * velocity + 9.81 * torch.sin(theta) - wind * torch.sin(1.5 * t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--warmup', type=int, default=2000, help='initial data-only training steps')
    parser.add_argument('--points', type=int, default=40, help='number of noisy observations')
    parser.add_argument('--collocation', type=int, default=128)
    parser.add_argument('--noise', type=float, default=0.03, help='Gaussian noise standard deviation in radians')
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--true-damping', type=float, default=0.25, help='synthetic damping in inverse seconds')
    parser.add_argument('--wind', type=float, default=0.0, help='simulated sinusoidal wind acceleration amplitude in rad/s²')
    parser.add_argument('--model-wind', type=float, default=0.0, help='wind amplitude assumed in the PINN equation')
    parser.add_argument('--output-prefix', default='', help='prefix for output files (letters, digits, hyphen, underscore)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.warmup < 0 or args.points < 2 or args.collocation < 2 or args.width < 1 or args.duration <= 0 or args.noise < 0 or args.true_damping < 0 or args.lr <= 0:
        parser.error('epochs, width, duration and lr must be positive; points and collocation >= 2; noise and damping >= 0')
    if args.wind < 0 or args.model_wind < 0 or (args.output_prefix and not all(c.isalnum() or c in "-_" for c in args.output_prefix)):
        parser.error("wind amplitudes must be nonnegative and output prefix must contain only letters, digits, hyphen or underscore")
    prefix = args.output_prefix + "_" if args.output_prefix else ""
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    times, observations, dense_times, truth = generate_data(args.seed, args.points, args.noise, args.duration, args.true_damping, args.wind)
    t_data = torch.tensor(times[:, None], dtype=torch.float32)
    y_data = torch.tensor(observations[:, None], dtype=torch.float32)
    # Collocation points enforce the equation between observations.
    t_physics = torch.linspace(0, args.duration, args.collocation).reshape(-1, 1)
    model = PendulumPINN(args.duration, args.width)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    for epoch in range(1, args.warmup + args.epochs + 1):
        optimizer.zero_grad()
        data_loss = (model(t_data) - y_data).square().mean()
        physics_loss = residual(model, t_physics, args.model_wind).square().mean()
        # Learn the oscillation first, then enforce the physical equation.
        loss = data_loss if epoch <= args.warmup else data_loss + physics_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 500 == 0 or epoch == args.warmup or epoch == args.warmup + args.epochs:
            history.append([epoch, loss.item(), data_loss.item(), physics_loss.item(), model.damping.item()])
            print(f'epoch {epoch:5d} loss={loss.item():.6f} data={data_loss.item():.6f} physics={physics_loss.item():.6f} b={model.damping.item():.4f}', flush=True)

    # The clean trajectory is reserved for evaluation, not training.
    with torch.no_grad():
        prediction = model(torch.tensor(dense_times[:, None], dtype=torch.float32)).numpy().ravel()
    damping = model.damping.item()
    rmse = float(np.sqrt(np.mean((prediction - truth)**2)))
    for folder in ('data', 'results', 'visualization'):
        (ROOT / folder).mkdir(exist_ok=True)
    np.savetxt(ROOT / 'data' / f'{prefix}observations.csv', np.column_stack((times, observations)), delimiter=',', header='time_s,angle_rad', comments='')
    np.savetxt(ROOT / 'results' / f'{prefix}predictions.csv', np.column_stack((dense_times, truth, prediction)), delimiter=',', header='time_s,true_angle_rad,predicted_angle_rad', comments='')
    np.savetxt(ROOT / 'results' / f'{prefix}training_history.csv', history, delimiter=',', header='epoch,total_loss,data_loss,physics_loss,damping', comments='')
    metrics = {'true_damping_per_s': args.true_damping, 'learned_damping_per_s': damping, 'trajectory_rmse_rad': rmse, 'config': vars(args)}
    (ROOT / 'results' / f'{prefix}metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    torch.save({'state_dict': model.state_dict(), 'duration': args.duration, 'width': args.width}, ROOT / 'results' / f'{prefix}model.pt')
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(dense_times, truth, label='Clean simulated trajectory')
    ax.plot(dense_times, prediction, '--', label='PINN prediction')
    ax.scatter(times, observations, s=18, color='black', alpha=.6, label='Noisy observations')
    ax.set(xlabel='Time (s)', ylabel='Angle (rad)', title=f'Wind: {args.wind:.2f}; assumed: {args.model_wind:.2f} rad/s² | damping: {damping:.3f} /s')
    ax.legend(); fig.tight_layout()
    fig.savefig(ROOT / 'visualization' / f'{prefix}pendulum_fit.png', dpi=160)
    plt.close(fig)
    print(f'Trajectory RMSE: {rmse:.5f} rad; learned damping: {damping:.5f} /s')


if __name__ == '__main__':
    main()
