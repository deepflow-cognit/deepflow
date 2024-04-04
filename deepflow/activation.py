
"""
`deepflow.activation()`
----

contains activation functions nessesary to create layers, or anything else
functions:

- `deepflow.activation.sigmoid()`
- `deepflow.activation.relu()`
- `deepflow.activation.elu()`
- `deepflow.activation.linear()`
- `deepflow.activation.mish()`
- `deepflow.activation.tanh()`
- `deepflow.activation.swish()`
- `deepflow.activation.forward()`
- `deepflow.activation.softmax()`


"""

import package.src.deepflow as deepflow

    
    
def sigmoid(X):
    """
    `deepflow.activation.sigmoid()`
    ----
    `X: input data`

    Applies the sigmoid activation function to the input data.

    Returns:
        The output data after applying the sigmoid function.
    """
    return 1 / (1 + np.exp(-X))


def relu(X):
    """
    `deepflow.activation.ReLU()`
    ----
    `X: input data`

    Applies the ReLU activation function to the input data.

    Returns:
        The output data after applying the ReLU function.
    """
    return np.maximum(0, X)  # Maximum of 0 and the input


def elu( X, alpha=1.0):
    """
    `deepflow.activation.elu()`
    ----
    `X: input data`
    `alpha: alpha parameter for the ELU function (default: 1.0)`

    Applies the ELU activation function to the input data.

    Returns:
        The output data after applying the ELU function.
    """
    return np.where(X <= 0, alpha * (np.exp(X) - 1), X)


def linear(X):
    """
    `deepflow.activation.linear()`
    ----
    `X: input data`

    Applies the linear activation function to the input data (identity function).

    Returns:
        The unmodified input data.
    """
    return X


def mish( X):
    """
    `deepflow.activation.mish()`
    ----
    `X: input data`

    Applies the Mish activation function to the input data.

    Returns:
        The output data after applying the Mish function.
    """
    return X * np.tanh(np.log1p(np.exp(X)))  # Mish formula


def tanh(X):
    """
    `deepflow.activation.tanh()`
    ----
    `X: input data`

    Applies the hyperbolic tangent (tanh) activation function to the input data.

    Returns:
        The output data after applying the tanh function.
    """
    return np.tanh(X)

def softmax(x):
    """
    `deepflow.activation.softmax`
    -----
    
    Computes the softmax of the input vector.

    Args:
        x: Input vector (numpy array).

    Returns:
        Softmax output vector (numpy array).
    """
    # Exponentiate the input values
    exp_x = np.exp(x)

    # Avoid potential overflow by calculating the sum of exponentials first
    sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)

    # Normalize by dividing with the sum to get probabilities
    return exp_x / sum_exp_x


def swish(X):
    """
    `deepflow.activation.swish()`
    ----
    `X: input data`

    Applies the Swish activation function to the input data.

    Returns:
        The output data after applying the Swish function.
    """
    return X * sigmoid(X)  # X * sigmoid(X)

@classmethod
def forward(X, activation_func="sigmoid"):
    """
    `deepflow.activation.forward()`
    ----
    `X: input data`
    `activation_func: string (optional, defaults to "sigmoid")`

    Performs forward propagation through two layers, applying the specified activation function after the first layer.

    Returns:
        The output of the second layer.
    """

    activation_map = {
            "sigmoid": sigmoid(),
            "relu": relu,
            "swish": swish,
            "elu": elu,
            
            "tanh": tanh,
            "mish": mish,
            "linear": linear,
            "softmax": softmax,
        }
    activation_func = activation_map.get(activation_func, sigmoid)  # Default to sigmoid if not specified

    layer1 = np.dot(X, deepflow.layers.weights1) + deepflow.layers.biases1
    layer1 = activation_func(layer1)  # Apply chosen activation
    output = np.dot(layer1, deepflow.layers.weights2) + deepflow.layers.biases2
    return output


def backward(X, y, output, activation_func="sigmoid"):
    """
    `deepflow.activation.backward()`
    ----
    `X: input data`
    `y: true labels`
    `output: output from the forward propagation`
    `activation_func: string (optional, defaults to "sigmoid")`

    Performs backward propagation using the output of the forward propagation.
    Adjusts the weights and biases based on the error rate.

    Returns:
        The gradients of the weights and biases.
    """

    activation_map = {
            "sigmoid": sigmoid(),
            "relu": relu,
            "swish": swish,
            "elu": elu,
            
            "tanh": tanh,
            "mish": mish,
            "linear": linear,
            "softmax": softmax,
        }
    activation_derivative = activation_map.get(activation_func, sigmoid)

    # Calculate the error
    error = y - output
    d_output = error * activation_derivative(output)

    # Calculate the gradient for weights2 and biases2
    layer1 = np.dot(X, deepflow.layers.weights1) + deepflow.layers.biases1
    layer1 = sigmoid(layer1)  # or your chosen activation function
    d_weights2 = np.dot(layer1.T, d_output)
    d_biases2 = np.sum(d_output, axis=0, keepdims=True)

    # Calculate the error for layer1
    d_layer1 = np.dot(d_output, deepflow.layers.weights2.T) * activation_derivative(layer1)

    # Calculate the gradient for weights1 and biases1
    d_weights1 = np.dot(X.T, d_layer1)
    d_biases1 = np.sum(d_layer1, axis=0, keepdims=True)

    # Update the weights and biases
    weights1 += d_weights1
    biases1 += d_biases1
    weights2 += d_weights2
    biases2 += d_biases2

    return d_weights1, d_biases1, d_weights2, d_biases2

if __name__ == "__main__":
    import numpy as np