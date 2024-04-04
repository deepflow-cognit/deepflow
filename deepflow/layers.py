
"""

    `deepflow.layers()`
    ----
    contains nessesary functions for creating input, hidden and output layers.
    
    functions:
    
    - `deepflow.layers.layer()`
    - `deepflow.layers.activation()`
    - `deepflow.layers.dense()`
    - `deepflow.layers.flatten()`
    - `deepflow.layers.dropout()`
    """
    
global leng
global mask
    
leng=0
mask=" "
from deepflow import activation
import numpy as np

if __name__ == "__main__":
    input_size = 0
    hidden_size = 0
    output_size = 0


activated_output = ""


def layer(input_size, hidden_size, output_size) -> None:
    """

    `deepflow.layers.layer()`
    ----
    `input_size: input neurons`
    `hidden_size: hidden neurons`
    `output_size: output neurons`
    """
    
    
    str(input_size)
    str(hidden_size)
    str(output_size)
    
    # Initialize weights and biases with random values
    weights1 = np.random.randn(input_size, hidden_size)
    biases1 = np.zeros((hidden_size,))
    weights2 = np.random.randn(hidden_size, output_size)
    biases2 = np.zeros((output_size,))
    

def dense(input_size, output_size=1, activation="relu") -> None:
    """
    `deepflow.layers.denseLayer()`
    ----
    Initializes the dense layer.

    Args:
        input_size (int): The number of inputs to the layer.
        output_size (int): The number of outputs from the layer.
        activation (str, optional): The activation function to use. Defaults to "relu".
    """
    # Initialize weights and biases with appropriate distribution (e.g., Xavier initialization)
    weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
    biases = np.zeros(output_size)
    # Store chosen activation function
    activation = None

  
def flatten(input_shape):
    """
    `deepflow.layers.flatten()`
    ----
    
    Performs the flattening operation.

    Args:
        X (np.ndarray): The input data.

    Returns:
        np.ndarray: The flattened output.
    """
    # Reshape the input data to a single dimension
    
    if isinstance(input_shape, tuple):
        # Do something if the variable is a tuple
        flatten = np.ravel(input_shape)
        return flatten
    elif isinstance(input_shape, np.ndarray):  # Check for NumPy array (optional)
        # Do something if the variable is a NumPy array
        flatten = input_shape.flatten
        return flatten
    else:
        # Handle other data types (optional)
        print("deepflow.layers.flatten() - variable is neither a tuple nor a NumPy array:", type(input_shape))
    
    
    flatten = np.ravel(input_shape)
    return flatten


def dropout(rate,input_shape):
    """
    `deepflow.layers.dropou=t()`
    -----
    Implements a Dropout layer during training.
    Args:
        X: Probability of keeping a neuron (1.0 - dropout rate).
        input_shape: array data 

    Returns:
        Output data with dropout applied (numpy array).
    """
    leng = 0.0
    mask = 0
    
    float(leng)
    
    if isinstance(input_shape, tuple):
        leng = 1
    elif isinstance(input_shape, np.ndarray):  # Check for NumPy array (optional)
        leng = 0
    else:
        print("deepflow.layers.dropout() - variable is neither a tuple nor a NumPy array:")

    if leng == 0:
        # Generate a random mask with values 0 or 1
        mask = np.random.rand(*input_shape.shape) < rate
    elif leng == 1:
        mask = np.random.rand(np.ndim(input_shape)) < rate

    # Apply the mask by multiplying with the input
    output = rate * mask

    # Invert dropout for training stability (optional)
    # Scaled output to maintain expected value during backpropagation
    output /= rate

    return output


def activation(activation,X):
    
    """
    `deepflow.layers.activation()`
    ----
    Applies a specified activation function to the input data.

Args:
    activation (str): The name of the activation function to use.
        Supported options include:
        
            - "sigmoid"
            - "ReLU"
            - "tanh"
            - "elu" (Exponential Linear Unit)
            - "mish" (Mish activation)
            - "linear"
            - "swish" (Swish activation)
    X (numpy.ndarray): The input data.

Returns:
    numpy.ndarray: The input data after applying the specified activation function.

Raises:
    ValueError: If an unsupported activation function is provided.
    """
    
    if activation == "sigmoid":
        activation.sigmoid(X)
    elif activation == "relu":
        activation.relu(X)
    elif activation == "tanh":
        activation.tanh(X)
    elif activation == "elu":
        activation.elu(X)
    elif activation == "mish":
        activation.tanh(X)
    elif activation == "linear":
        activation.linear(X)
    elif activation == "swish":
        activation.swish(X)
    elif activation == "softmax":
        activation.softmax(X)
    else:
        raise ValueError("Unsupported activation function: {}".format(activation))
