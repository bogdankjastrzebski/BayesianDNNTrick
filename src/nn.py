
# Imports
import jax 
import jax.numpy as np
import jax.scipy as sp


def Dense(input_dim, output_dim, activation_function):
    def init(key):
        weight = jax.random.normal(key, shape=(input_dim, output_dim))/input_dim # /output_dim
        bias = np.zeros((output_dim,))
        return weight, bias
    def forward(params, X):
        weight, bias = params
        return activation_function(X @ weight + bias)
    return init, forward

def Sequential(*args):
    inits, forwards = zip(*args)
    def init(key):
        return [
            init(key)
            for init, key in zip(
                inits, 
                jax.random.split(key, len(inits))
            )
        ]
    def forward(params, X):
        for forward, params in zip(forwards, params):
            X = forward(params, X)
        return X
    return init, forward

def Residual(layer):
    init, forward = layer
    def residual_forward(params, X):
        return forward(params, X) + X
    return init, residual_forward 

def ExpandSqueeze(inout_dim, hidden_dim, activation_function=lambda x: x):
    return Residual(Sequential(
        Dense(inout_dim, hidden_dim, jax.nn.relu),
        Dense(hidden_dim, hidden_dim, jax.nn.relu),
        Dense(hidden_dim, inout_dim, lambda x: x),
    ))
