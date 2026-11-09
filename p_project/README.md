# Discovering pendulum friction with a physics-informed neural network

This example fits sparse, noisy angular measurements while learning the damping coefficient of a nonlinear pendulum. The simulated equation is

`theta''(t) + b theta'(t) + (g/L) sin(theta(t)) = 0`,

with `g/L = 9.81 s^-2`, initial angle 1 radian, initial angular velocity 0, and default true damping `b = 0.25 s^-1`. The model knows the equation and `g/L`, but not the true damping or clean trajectory. This is a normalized equation: `b` is the damping rate, not an unnormalized torque coefficient.

## Run

Use Python 3.10 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/pendulum_pinn.py
```

Run `python src/pendulum_pinn.py --help` for options such as noise, number of observations, duration, seed, and training epochs. The default experiment uses 40 observations over 6 seconds and 2,000 data-only warmup steps followed by 6,000 physics-informed steps. Results depend on the random seed, noise, and optimization settings; the learned coefficient is an estimate, not guaranteed to match exactly.

## How it works

1. SciPy `solve_ivp` generates the clean trajectory; independent Gaussian noise is added at observation times.
2. A three-hidden-layer tanh network maps time to angle. Its trainable damping is constrained to be positive with softplus.
3. PyTorch automatic differentiation computes the first and second time derivatives. The loss is mean squared observation error plus mean squared ODE residual at 128 collocation times. No clean trajectory or true damping is used in the loss.
4. Adam first fits the observations to establish the oscillation, then jointly trains the network and damping coefficient with the physics loss. The simulated clean trajectory is used afterward only to calculate an evaluation RMSE and make the plot.

Every run overwrites `data/observations.csv`, `results/predictions.csv`, `results/training_history.csv`, `results/metrics.json`, `results/model.pt`, and `visualization/pendulum_fit.png`. The model checkpoint contains its state dictionary, width, and time range. Initial conditions are implicit in the measurements rather than imposed as exact constraints.

## Wind experiment

A simple, deterministic wind torque adds `A sin(1.5 t)` to the right-hand side of the equation, where `A` is angular acceleration in rad/s². This is an illustrative periodic disturbance, not a realistic gust model. `--wind` controls the simulated disturbance; `--model-wind` controls what the PINN assumes is known. The frequency is fixed and known. For a reproducible comparison using the same noisy observations, run:

```sh
python src/pendulum_pinn.py --wind 0.3 --model-wind 0 --epochs 16000 --output-prefix wind_unmodeled
python src/pendulum_pinn.py --wind 0.3 --model-wind 0.3 --epochs 16000 --output-prefix wind_modeled
```

The prefix keeps the two sets of outputs separate. In the included seed-42 run, ignoring the wind gave a clean-trajectory RMSE of 0.0330 rad and learned damping 0.2467 s⁻¹; including the exact wind gave 0.0080 rad and 0.2522 s⁻¹ (true 0.25 s⁻¹). Both use 40 noisy measurements. The wind-aware model is given the true forcing, so this compares equation mismatch against known forcing; it does not demonstrate discovery of unknown wind or establish performance across different seeds and disturbances. RMSE uses the clean simulated trajectory, which would not normally be available for real sensor data.
