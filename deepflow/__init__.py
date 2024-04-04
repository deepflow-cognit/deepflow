"""
`deepflow`
=====

deepflow is a neural networking package made originally in python. it can be used by Initializing weights and biases with random values
deepflow also helps with training the neural network with it's `train_data()` function.

sub-modules:

- `deepflow.layers()`
- `deepflow.activation()`
- `deepflow.optimisers()`
- `deepflow.losses()`
- `deepflow.dataset()`
- `deepflow.sequential`

"""

__version__ = "deepflow package v1.6"

def _DeepflowLazyLoader(func):
  """
  Decorator to lazily load a property.

  Args:
      func: The function that calculates the property value.

  Returns:
      A property object that calls the decorated function only on first access.
  """
  attr_name = f"_{func.__name__}"  # Create private attribute name

  @property
  def wrapper(self):
    if not hasattr(self, attr_name):
      setattr(self, attr_name, func(self))
    return getattr(self, attr_name)

  return wrapper


