
import torch
import torchvision as tv

import jax 
import jax.numpy as np
import jax.scipy as sp

from tqdm import tqdm, trange
from jax import value_and_grad, grad, jit, vmap
from jax.example_libraries import optimizers

import pickle

from nn import * 


# Constraints
def es_constraint(params):
    (w0, _), (w1, _), (w2, _) = params
    return (
        jax.nn.relu(np.linalg.norm(w0, ord=2) - 0.95) +
        jax.nn.relu(np.linalg.norm(w1, ord=2) - 0.95) +
        jax.nn.relu(np.linalg.norm(w2, ord=2) - 0.95)
    ) 
    
def constraint(params):
    return sum(es_constraint(p) for p in params)

# Loss Function
def log_density(params, X):
    
    Z, pullback = jax.vjp(lambda X: forward(params, X), X)
    
    basis = np.eye(Z.shape[1], dtype=Z.dtype)[:,np.newaxis,:].repeat(Z.shape[0], axis=1)
    jac = vmap(pullback)(basis)[0].transpose((1,0,2))
    
    return sp.stats.norm.logpdf(Z).sum(axis=1) + np.log(np.abs(jax.vmap(np.linalg.det)(jac)))

def loss(params, X, C):
    weights = [p[0] for p in params]
    return -log_density(params, X).sum() + C*constraint(params)


# Data Preparation
ds = tv.datasets.MNIST("../data")
def collate_fn(batch):
    return np.c_[[np.array(x) for x, _ in batch]].reshape(-1,784).astype(np.float32) / 255 - 0.5
dl = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=collate_fn, shuffle=True)

# Model
n = 32
init, forward = Sequential(
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
    ExpandSqueeze(784, n),
)

key = jax.random.PRNGKey(5022023)
params = init(key)
losses = []

learning_rate=0.001
opt_init, opt_update, get_params = optimizers.adam(learning_rate)
opt_state = opt_init(params)

key = jax.random.PRNGKey(9022023)
EPOCHS = 1
C = 1000.0
jitvgloss = jit(lambda params, X, C: jax.value_and_grad(lambda params: loss(params, X, C))(params))

for epoch in (pbar := trange(EPOCHS)):
    
    for X in dl:
        value, grads = jitvgloss(get_params(opt_state), X, C)
        opt_state = opt_update(epoch, grads, opt_state)

        losses.append(value)
        pbar.set_description(f"{value:0.03f}")
        break 
    break
        

with open('model.pickle', 'wb') as f:
    pickle.dump((get_params(opt_state), losses), f)
