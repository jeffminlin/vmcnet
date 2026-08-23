# Carbon minSR and EMA Nyström Comparison

## Goal

Test whether the regularization-coupled streaming Nyström metric improves a
real-scale VMCNet minSR calculation. Use the carbon-atom settings and tuned
minSR hyperparameters from Figure 2 of the SPRING paper, but start directly
from the seeded VMCNet initialization instead of a KFAC preliminary
optimization.

Run two matched single-seed calculations concurrently:

- `minSR`: ordinary VMCNet minSR;
- `minSR-Nyström-EMA-adapt`: minSR in the smart regularization-coupled
  streaming Nyström metric.

## Shared Physical And VMC Settings

Use one carbon ion at the origin:

```text
ion_pos = ((0.0, 0.0, 0.0),)
ion_charges = (6.0,)
nelec = (4, 2)
```

Use the default VMCNet 16-determinant dense FermiNet and seed `0`. Do not load
a preliminary-optimization or pretraining checkpoint.

Use the paper's training and inference settings:

```text
training walkers = 1000
training burn-in = 5000
MCMC steps between updates = 10
training iterations = 100000
evaluation walkers = 2000
evaluation burn-in = 5000
MCMC steps between measurements = 10
evaluation iterations = 20000
```

Retain the default mean-centered local-energy clipping threshold `5`, dynamic
proposal-width adjustment, float32 arithmetic, and single-device execution.

Use the tuned carbon minSR hyperparameters from the paper for both methods:

```text
learning_rate = 0.1
schedule_type = inverse_time
learning_decay_rate = 1e-4
lambda = 1e-3
mu = 0
constrain_norm = true
norm_constraint = 1e-3
```

Here `mu = 0` makes both optimizers minSR methods rather than SPRING methods.

## Ordinary minSR

Use VMCNet optimizer `spring`, with `damping = lambda`.

For the centered, sample-normalized log-derivative matrix $O_t$ and centered
local-energy vector $e_t$, solve

$$
(O_t O_t^T + \lambda I)y_t=e_t,
\qquad d_t=O_t^T y_t.
$$

## Smart EMA Nyström minSR

Use VMCNet optimizer `spring_nystrom` with:

```text
nystrom_rank = 100
nystrom_ema_decay = 0.999
nystrom_warmup_steps = 0
nystrom_phasein_steps = 0
collect_during_warmup = true
metric_shift_strategy = regularization_coupled
metric_normalization = none
metric_identity_shift = 1
eigenvalue_floor = 1e-8
```

With fixed Gaussian probe matrix $\Omega$, update the uncorrected Fisher sketch
as

$$
Y_t=0.999Y_{t-1}+0.001O_t^T(O_t\Omega),
\qquad Y_0=0.
$$

Recover the rank-100 Nyström approximation
$\widehat F_t=U_t\Lambda_tU_t^T$ and use

$$
B_t=I+\frac{\widehat F_t}{\lambda}.
$$

The uncorrected EMA naturally phases the metric in from the identity, so do
not add a separate warmup or phase-in schedule. Apply $B_t^{-1}$ in the
weighted minSR system

$$
(O_tB_t^{-1}O_t^T+\lambda I)y_t=e_t,
\qquad d_t=B_t^{-1}O_t^Ty_t.
$$

The same $\lambda=10^{-3}$ is intentionally used for the sample-space
Tikhonov damping and the scale of the explicit Fisher approximation. The
`eigenvalue_floor` is separate sketch-level numerical stabilization only.

## Cluster Execution

Use persistent Slurm batch jobs on the `savio3_gpu` partition with one
`GTX2080TI` GPU and two CPU cores per method. Use the
`codex_preconditioning` environment with a purged module environment and
disabled Weights & Biases logging.

Store all logs, checkpoints, metrics, and evaluation samples below

```text
/global/scratch/users/ggoldshlager/codex/preconditioning/carbon_fig2_nystrom/
```

Save regular checkpoints every `5000` training iterations. Before submitting
the full jobs, run both configurations for two training iterations at the full
walker count and network size, with evaluation disabled. Proceed to the full
runs only if both batch smoke tests complete successfully on a 2080 Ti.

## Analysis

For each completed run, retain the raw training energy, variance, acceptance,
Nyström diagnostic, checkpoint, and evaluation files produced by VMCNet.
Compare training trajectories using 10,000-iteration moving averages, as in
the paper, and report evaluation energy and uncertainty from the 20,000-step
inference phase. Use the carbon benchmark energy `-37.8450` Ha for the plotted
absolute energy error.
