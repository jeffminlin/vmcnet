# >>> from optax library

from typing import Any, Optional

from optax._src import base
from optax._src import combine
from optax._src import transform
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate

def sgd(
    learning_rate: ScalarOrSchedule,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  """A canonical Stochastic Gradient Descent optimizer.

  This implements stochastic gradient descent. It also includes support for
  momentum, and nesterov acceleration, as these are standard practice when
  using stochastic gradient descent to train deep neural networks.

  References:
    Sutskever et al, 2013: http://proceedings.mlr.press/v28/sutskever13.pdf

  Args:
    learning_rate: A fixed global scaling factor.
    momentum: Decay rate used by the momentum term, when it is set to `None`,
      then momentum is not used at all.
    nesterov: Whether Nesterov momentum is used.
    accumulator_dtype: Optional `dtype` to be used for the accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation`.
  """
  return combine.chain(
      (transform.trace(decay=momentum, nesterov=nesterov,
                       accumulator_dtype=accumulator_dtype)
       if momentum is not None else base.identity()),
      _scale_by_learning_rate(learning_rate)
  )

# from optax library <<<