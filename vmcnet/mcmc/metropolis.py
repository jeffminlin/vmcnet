"""Proposal and acceptance fns for Metropolis-Hastings Markov-Chain Monte Carlo."""
import logging
from typing import Callable, Tuple, cast

import chex
import jax
import jax.numpy as jnp

import vmcnet.utils as utils
from vmcnet.utils.typing import Array, D, P, PRNGKey

MetropolisStep = Callable[[P, D, PRNGKey], Tuple[chex.Numeric, D, PRNGKey]]
WalkerFn = MetropolisStep[P, D]
BurningStep = Callable[[P, D, PRNGKey], Tuple[D, PRNGKey]]


def make_metropolis_step(
    proposal_fn: Callable[[P, D, PRNGKey], Tuple[D, PRNGKey]],
    acceptance_fn: Callable[[P, D, D], Array],
    update_data_fn: Callable[[D, D, Array], D],
) -> MetropolisStep[P, D]:
    """Factory to create a function which takes a single metropolis step.

    Following Metropolis-Hastings Markov Chain Monte Carlo, a transition from one data
    state to another is split into proposal and acceptance. When used in a Metropolis
    routine to approximate a stationary distribution P, the proposal and acceptance
    functions should satisfy detailed balance, i.e.,

        proposal_prob_ij * acceptance_ij * P_i = proposal_prob_ji * acceptance_ji * P_j,

    where proposal_prob_ij is the likelihood of proposing the transition from state i to
    state j, acceptance_ij is the likelihood of accepting a transition from state i
    to state j, and P_i is the probability of being in state i.

    Args:
        proposal_fn (Callable): proposal function which produces new proposed data. Has
            the signature (params, data, key) -> proposed_data, key
        acceptance_fn (Callable): acceptance function which produces a vector of numbers
            used to create a mask for accepting the proposals. Has the signature
            (params, data, proposed_data) -> Array: acceptance probabilities
        update_data_fn (Callable): function used to update the data given the original
            data, the proposed data, and the array mask identifying which proposals to
            accept. Has the signature
            (data, proposed_data, mask) -> new_data

    Returns:
        Callable: function which takes in (data, params, key) and outputs
        (mean acceptance probability, new data, new jax PRNG key split from previous
        one)
    """

    def metrop_step_fn(
        params: P, data: D, key: PRNGKey
    ) -> Tuple[chex.Numeric, D, PRNGKey]:
        """Take a single metropolis step."""
        key, subkey = jax.random.split(key)
        proposed_data, key = proposal_fn(params, data, key)
        accept_prob = acceptance_fn(params, data, proposed_data)
        move_mask = cast(
            Array,
            jax.random.uniform(subkey, shape=accept_prob.shape) < accept_prob,
        )
        new_data = update_data_fn(data, proposed_data, move_mask)

        return jnp.mean(accept_prob), new_data, key

    return metrop_step_fn


def walk_data(
    nsteps: int,
    params: P,
    data: D,
    key: PRNGKey,
    metrop_step_fn: MetropolisStep[P, D],
) -> Tuple[chex.Numeric, D, PRNGKey]:
    """Take multiple Metropolis-Hastings steps.

    This function is roughly equivalent to:
    ```
    accept_sum = 0.0
    for _ in range(nsteps):
        accept_prob, data, key = metropolis_step_fn(data, params, key)
        accept_sum += accept_prob
    return accept_sum / nsteps, data, key
    ```

    but has better tracing/pmap behavior due to the use of jax.lax.scan instead of a
    python for loop. See :func:`~vmcnet.train.vmc.take_metropolis_step`.

    Args:
        nsteps (int): number of steps to take
        data (pytree-like): data to walk (update) with each step
        params (pytree-like): parameters passed to proposal_fn and acceptance_fn, e.g.
            model params
        key (PRNGKey): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)

    Returns:
        (jnp.chex.Numeric, pytree-like, PRNGKey): acceptance probability, new data,
            new jax PRNG key split (possibly multiple times) from previous one
    """

    def step_fn(carry, x):
        del x
        accept_prob, data, key = metrop_step_fn(params, carry[1], carry[2])
        return (carry[0] + accept_prob, data, key), None

    out = jax.lax.scan(step_fn, (0.0, data, key), xs=None, length=nsteps)
    accept_sum, data, key = out[0]
    return accept_sum / nsteps, data, key


def make_jitted_burning_step(
    metrop_step_fn: MetropolisStep[P, D],
    apply_pmap: bool = True,
) -> BurningStep[P, D]:
    """Factory to create a burning step, which is an optionally pmapped Metropolis step.

    This provides the functionality to optionally apply jax.pmap to a single Metropolis
    step. Only one step is traced so that the first burning step is traced but
    subsequent steps are properly jit-compiled. The acceptance probabilities (which
    typically don't mean much during burning) are thrown away.

    For more about the Metropolis step itself, see
    :func:`~vmcnet.mcmc.metropolis.make_metropolis_step`.

    Args:
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept prob, new data, new key)
        apply_pmap (bool, optional): whether to apply jax.pmap to the burning step. If
            False, applies jax.jit. Defaults to True.

    Returns:
        Callable: function with signature
            (data, params, key) -> (data, key),
        with jax.pmap optionally applied if apply_pmap is True.
    """

    def burning_step(params: P, data: D, key: PRNGKey) -> Tuple[D, PRNGKey]:
        _, data, key = metrop_step_fn(params, data, key)
        return data, key

    if not apply_pmap:
        return jax.jit(burning_step)

    return utils.distribute.pmap(burning_step)


def make_jitted_walker_fn(
    nsteps: int,
    metrop_step_fn: MetropolisStep[P, D],
    apply_pmap: bool = True,
) -> WalkerFn[P, D]:
    """Factory to create a function which takes multiple Metropolis steps.

    This provides the functionality to optionally apply jax.pmap to a jax.lax.scan loop
    of multiple metropolis steps. A typical use case would be to run this function
    between parameter updates in a VMC loop. An accumulated mean acceptance probability
    statistic is returned from this walker function.

    See :func:`~vmcnet.train.vmc.vmc_loop` for usage.

    Args:
        nsteps (int): number of metropolis steps to take in each call
        metrop_step_fn (Callable): function which does a metropolis step. Has the
            signature (data, params, key) -> (mean accept probl, new data, new key)
        apply_pmap (bool, optional): whether to apply jax.pmap to the walker function.
            If False, applies jax.jit. Defaults to True.

    Returns:
        Callable: funciton with signature
            (params, data, key) -> (mean accept prob, new data, new key)
        with jax.pmap optionally applied if pmapped is True, and jax.jit applied if
        apply_pmap is False.
    """

    def walker_fn(params: P, data: D, key: PRNGKey) -> Tuple[chex.Numeric, D, PRNGKey]:
        accept_ratio, data, key = walk_data(nsteps, params, data, key, metrop_step_fn)
        accept_ratio = utils.distribute.pmean_if_pmap(accept_ratio)
        return accept_ratio, data, key

    if not apply_pmap:
        return jax.jit(walker_fn)

    pmapped_walker_fn = utils.distribute.pmap(walker_fn)

    def pmapped_walker_fn_with_single_accept_ratio(
        params: P, data: D, key: PRNGKey
    ) -> Tuple[chex.Numeric, D, PRNGKey]:
        accept_ratio, data, key = pmapped_walker_fn(params, data, key)
        accept_ratio = utils.distribute.get_first(accept_ratio)
        return accept_ratio, data, key

    return pmapped_walker_fn_with_single_accept_ratio


def burn_data(
    burning_step: BurningStep[P, D],
    nsteps_to_burn: int,
    params: P,
    data: D,
    key: PRNGKey,
) -> Tuple[D, PRNGKey]:
    """Repeatedly apply a burning step.

    Args:
        burning_step (BurningStep): function which does a burning step. Has the
            signature (data, params, key) -> (new data, new key)
        nsteps_to_burn (int): number of times to call burning_step
        data (pytree-like): initial data
        params (pytree-like): parameters passed to the burning step
        key (PRNGKey): an array with shape (2,) representing a jax PRNG key passed
            to proposal_fn and used to randomly accept proposals with probabilities
            output by acceptance_fn

    Returns:
        (pytree-like, PRNGKey): new data, new key
    """
    logging.info("Burning data for %d steps", nsteps_to_burn)
    for _ in range(nsteps_to_burn):
        data, key = burning_step(params, data, key)
    return data, key


def gaussian_proposal(
    positions: Array, std_move: chex.Scalar, key: PRNGKey
) -> Tuple[Array, PRNGKey]:
    """Simple symmetric gaussian proposal in all positions at once.

    Args:
        positions (Array): original positions
        std_move (chex.Scalar): standard deviation of the moves
        key (PRNGKey): an array with shape (2,) representing a jax PRNG key

    Returns:
        (Array, Array): (new positions, new key split from previous)
    """
    key, subkey = jax.random.split(key)
    return positions + std_move * jax.random.normal(subkey, shape=positions.shape), key


def metropolis_symmetric_acceptance(
    amplitude: Array, proposed_amplitude: Array, logabs: bool = True
) -> Array:
    """Standard Metropolis acceptance ratio for a symmetric proposal function.

    The general Metropolis-Hastings choice of acceptance ratio for moves from state i to
    state j is given by

        accept_ij = min(1, (P_j * proposal_prob_ji) / (P_i * proposal_prob_ij)).

    When proposal_prob is symmetric (assumed in this function), this simply reduces to
    accept_ij = min(1, P_j / P_i). Some care is taken to avoid numerical overflow and
    division by zero.

    The inputs are wavefunction amplitudes psi or log(|psi|), so the probability P_i
    refers to |psi(i)|^2.

    Args:
        amplitude (Array): one-dimensional array of wavefunction amplitudes for
            the current state, or log wavefunction amplitudes if logabs is True
        proposed_amplitude (Array): one-dimensional array of wavefunction
            amplitudes for the proposed state, or log wavefunction amplitudes if logabs
            is True
        logabs (bool, optional): whether the provided amplitudes represent psi
            (logabs = False) or log|psi| (logabs = True). Defaults to True.

    Returns:
        Array: one-dimensional array of acceptance ratios for the Metropolis
        algorithm
    """
    if not logabs:
        prob_old = jnp.square(amplitude)
        prob_new = jnp.square(proposed_amplitude)
        ratio = prob_new / prob_old
        # safe division by zero
        ratio = jnp.where(
            jnp.logical_or(prob_old < prob_new, prob_old == 0.0),
            jnp.ones_like(ratio),
            ratio,
        )
        return ratio

    log_prob_old = 2.0 * amplitude
    log_prob_new = 2.0 * proposed_amplitude
    # avoid overflow if log_prob_new - log_prob_old is large
    return jnp.where(
        log_prob_new > log_prob_old,
        jnp.ones_like(log_prob_new),
        jnp.exp(log_prob_new - log_prob_old),
    )
