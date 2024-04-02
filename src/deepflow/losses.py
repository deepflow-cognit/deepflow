
"""
`deepflow.losses()`
----

contains mse and CE (cross entropy) loss calculators used for training neurons
functions:

- `deepflow.losses.mse()`
- `deepflow.losses.CE()` 
- `deepflow.losses.sce()`
"""
if __name__ == "__main__":
    import numpy as np

def mse(y_true, y_pred):
    """
    `deepflow.losses.mse()`
    ----
    Calculates the mean squared error between true and predicted values.

    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The mean squared error loss.
    """
    return np.mean((y_true - y_pred) ** 2)

def CE(y_true, y_pred, epsilon=1e-10):
    """
    `deepflow.losses.CE()`
    ----
    
    Calculates the cross-entropy loss between true target distribution and predicted probabilities.

    Args:
        y_true (np.ndarray): The one-hot encoded true target distribution.
        y_pred (np.ndarray): The predicted probabilities.
        epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

    Returns:
        float: The cross-entropy loss.
    """
def sce(y_true, y_pred, epsilon=1e-10):
    """
    Calculates the sparse categorical crossentropy loss.

    Args:
        y_true: Ground truth labels, an integer array of shape (num_samples,).
        y_pred: Predicted probabilities, a float array of shape (num_samples, num_classes).

    Returns:
        The average sparse categorical crossentropy loss across samples.
    """
    # Clip predictions to avoid overflow/underflow in log
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # One-hot encode the labels
    y_true = np.eye(y_pred.shape[1])[y_true]

    # Calculate the loss per sample
    loss = -np.sum(y_true * np.log(y_pred), axis=1)

    # Return the average loss

    # Clip predicted probabilities to avoid division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Calculate cross-entropy for each sample
    cross_entropy_loss = -np.sum(y_true * np.log(y_pred), axis=1)
    # Return average cross-entropy
    return np.mean(cross_entropy_loss)