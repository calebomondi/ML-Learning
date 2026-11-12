from typing import Callable, TypeVar, List, Iterator
import random

from scratch.linear_algebra import Vector, add, scalar_multiply


def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
    return (f(x + h) - f(x)) / h


def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
    """Returns the i-th partial difference quotient of f at v"""
    w = [v_j + (h if j == i else 0)    # add h to just the ith element of v
         for j, v_j in enumerate(v)]

    return (f(w) - f(v)) / h


def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001) -> Vector:
    return [partial_difference_quotient(f, v, i, h)
            for i in range(len(v))]


def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
    """Moves `step_size` in the `gradient` direction from `v`"""
    assert len(v) == len(gradient)
    step = scalar_multiply(step_size, gradient)
    return add(v, step)


def sum_of_squares_gradient(v: Vector) -> Vector:
    return [2 * v_i for v_i in v]


def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
    slope, intercept = theta
    predicted = slope * x + intercept    # The prediction of the model.
    error = (predicted - y)              # error is (predicted - actual)
    grad = [2 * error * x, 2 * error]    # gradient of the squared error
    return grad


T = TypeVar('T')  # this allows us to type "generic" functions


def minibatches(dataset: List[T],
                batch_size: int,
                shuffle: bool = True) -> Iterator[List[T]]:
    """Generates `batch_size`-sized minibatches from the dataset"""
    # Start indexes 0, batch_size, 2 * batch_size, ...
    batch_starts = [start for start in range(0, len(dataset), batch_size)]

    if shuffle: random.shuffle(batch_starts)  # shuffle the batches

    for start in batch_starts:
        end = start + batch_size
        yield dataset[start:end]


assert gradient_step([1, 2], [0.5, -1], 2) == [2, 0]
assert sum_of_squares_gradient([1, 2, 3]) == [2, 4, 6]
