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

from deepflow import activation,optimisers,losses
import sys,time,numpy as np

class sequential:
  """
  `deepflow.sequential([])`
  --------

  Creates a sequential container (and runs it)

  Args:
      first_arg: The first argument, which should be a list of code blocks or
                  a single code block to be executed.
      *args: Additional positional arguments to be passed to the code blocks.
      **kwargs: Additional keyword arguments to be passed to the code blocks.

  Returns:
      The result of the last executed code block, if any.

  fuctions:

  `deepflow.seqential.train_data()`
  """
  
  
  def __init__(self,layers_) -> None:
    layers_ = ""
    for item in layers_:
        if callable(item):
            item()  # Call the item if it's a function
    else:
        pass

  
  
  def evaluate(self, X, y, loss_calc):
      """
      `deepflow.sequential.evaluate()`
      --------------
      Evaluates the neural network performance on a given dataset.

      Args:
          X (numpy.ndarray): The input data for evaluation.
          y (numpy.ndarray): The target values (labels) for evaluation.
          layers_ (list): List of layer objects defining the network architecture.
          loss_calc (function): Function that calculates the loss between predicted and true values.

      Returns:
          float: The calculated loss on the evaluation data.
      """

      # Forward propagation
      y_pred = activation.forward(X, self.layers_)  # Use your custom forward function

      # Calculate loss
      loss = loss_calc(y, y_pred)

      return loss


  
  def train_data(self, optimiser="adam", X=None, y=None, layers_=[],loss_calc="mse", epochs=100, min_delta=0.001, patience=5):
      """
      `deepflow.sequential.train_data()`
      -----
      
      Trains the neural network using provided data with Adam optimizer and early stopping.

      Args:
          X (numpy.ndarray): The input data.
          y (numpy.ndarray): The target values (labels).
          layers (list): List of layer objects defining the network architecture.
          loss_calc (function): Function that calculates the loss between predicted and true values.
          epochs (int, optional): The maximum number of training epochs. Defaults to 100.
          min_delta (float, optional): Minimum change in loss to consider improvement. Defaults to 0.001.
          patience (int, optional): Number of epochs with no improvement to trigger early stopping. Defaults to 5.

      Returns:
          list: List of training losses for each epoch.
      """

      # Check if input data and target values have the same shape
      if X.shape != y.shape:
          raise ValueError("deepflow.sequential.train_data() - input data and target values must have the same shape.")

      # Training loop
      training_losses = []
      best_loss = np.inf
      epochs_no_improvement = 0
      for epoch in range(epochs):
          # Forward propagation
          y_pred = activation.forward(X,layers_)

          # Calculate loss
          loss = losses.loss_calc(y, y_pred)
          training_losses.append(loss)


          #animation = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
          animation = ["[==>                  ] 10%","[====>                ] 20%", "[======>              ] 30%", "[========>            ] 40%", "[==========>          ] 50%", "[============>        ] 60%", "[==============>     ] 70%", "[================>    ] 80%", "[==================>  ] 90%", "[====================>] 100% "]

          for i in range(len(animation)):
              time.sleep(0.2)
              sys.stdout.write("\r" + animation[i % len(animation)])
              sys.stdout.flush()
              print(f"\nEpoch: {epoch+1} - loss: {loss:.4f}")

          # Backpropagation using your custom function
          grads = activation.backward(X,y_pred,layers_)  # Use your custom backward function

          # Update weights and biases using Adam optimizer
          if optimiser == "adam":
              updated_params = optimisers.adam(
                  [layer.weights for layer in layers_] + [layer.biases for layer in layers_], grads)
          else:
              print("deepflow.optimisers.adam() - optimiser not supported.")
          
          for i, layer in enumerate(layers_):
              layer.weights = updated_params[2 * i]
              layer.biases = updated_params[2 * i + 1]