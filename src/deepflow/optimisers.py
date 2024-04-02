"""
`deepflow.optimisers()`
------

stores built in optimesers.
functions:

- `deepflow.optimiser.adam()`
"""

if __name__ == "__main__":
    import numpy as np
def adam(params, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    `deepflow.optimiser.adam()`
    ------
    
    Implements the Adam optimizer for parameter updates.

    Args:
        params: List of NumPy arrays containing model parameters.
        grads: List of NumPy arrays containing gradients for each parameter.
        learning_rate: Learning rate (default: 0.001).
        beta1: Decay rate for 1st moment estimate (default: 0.9).
        beta2: Decay rate for 2nd moment estimate (default: 0.999).
        epsilon: Small value to prevent division by zero (default: 1e-8).

    Returns:
        List of updated parameters (NumPy arrays).
    """

    # Initialize moment estimates (zeros)
    m_t = [np.zeros_like(p) for p in params]
    v_t = [np.zeros_like(p) for p in params]

    t = 0  # Timestep

    updated_params = []
    for p, g in zip(params, grads):
        t += 1

        # Update moving average of gradient (1st moment)
        m_t[0] = beta1 * m_t[0] + (1 - beta1) * g

        # Update moving average of squared gradient (2nd moment)
        v_t[0] = beta2 * v_t[0] + (1 - beta2) * g**2

        # Bias correction for 1st moment
        m_hat = m_t[0] / (1 - beta1**t)

        # Bias correction for 2nd moment
        v_hat = v_t[0] / (1 - beta2**t)

        # Update parameter with Adam formula
        p -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        updated_params.append(p)

    return updated_params


    
