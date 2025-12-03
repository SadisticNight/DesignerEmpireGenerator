# intrinsic_rnd.py — RND sobre features (JAX+Optax), con normalización EMA
from __future__ import annotations
import os
import jax, jax.numpy as jnp
import optax

def _he_init(key, n_in, n_out, scale=2.0):
    return jax.random.normal(key, (n_in, n_out), dtype=jnp.float32) * jnp.sqrt(jnp.float32(scale / max(1, n_in)))

def _init_mlp(key, d_in, d_h1, d_h2, d_out):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = _he_init(k1, d_in, d_h1); b1 = jnp.zeros((d_h1,), jnp.float32)
    W2 = _he_init(k2, d_h1, d_h2); b2 = jnp.zeros((d_h2,), jnp.float32)
    W3 = _he_init(k3, d_h2, d_out); b3 = jnp.zeros((d_out,), jnp.float32)
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}

@jax.jit
def _mlp_forward(p, x):
    if x.ndim == 1: x = x[None, :]
    h1 = jax.nn.relu(x @ p['W1'] + p['b1'])
    h2 = jax.nn.relu(h1 @ p['W2'] + p['b2'])
    y  = h2 @ p['W3'] + p['b3']
    return y  # (B, d_out)

@jax.jit
def _loss_and_se(pred_params, tgt_params, x):
    y_t = _mlp_forward(tgt_params, x)
    y_p = _mlp_forward(pred_params, x)
    se  = jnp.mean((y_p - y_t) ** 2, axis=-1)      # (B,)
    return jnp.mean(se), se

class RND:
    __slots__ = (
        'd_in','d_h1','d_h2','d_out',
        'pred','tgt','opt','opt_state',
        'ema_alpha','ema_mean','ema_sq','eps'
    )

    def __init__(self, obs_dim: int, seed: int = 0):
        self.d_in  = int(obs_dim)
        self.d_h1  = int(os.getenv('DEG_RND_H1', '128'))
        self.d_h2  = int(os.getenv('DEG_RND_H2', '128'))
        self.d_out = int(os.getenv('DEG_RND_OUT', '64'))
        lr = float(os.getenv('DEG_RND_LR', '1e-3'))
        self.ema_alpha = float(os.getenv('DEG_RND_EMA', '0.99'))
        self.ema_mean = jnp.array(0.0, jnp.float32)
        self.ema_sq   = jnp.array(1e-6, jnp.float32)
        self.eps = jnp.float32(1e-6)

        key = jax.random.PRNGKey(int(seed))
        k_pred, k_tgt = jax.random.split(key)
        self.pred = _init_mlp(k_pred, self.d_in, self.d_h1, self.d_h2, self.d_out)
        self.tgt  = _init_mlp(k_tgt,  self.d_in, self.d_h1, self.d_h2, self.d_out)
        # congelar target al no actualizar sus params
        self.opt = optax.adam(learning_rate=lr)
        self.opt_state = self.opt.init(self.pred)

    def _update_norm(self, x_scalar: jnp.ndarray) -> None:
        a = jnp.float32(self.ema_alpha)
        self.ema_mean = a * self.ema_mean + (1.0 - a) * x_scalar
        self.ema_sq   = a * self.ema_sq   + (1.0 - a) * (x_scalar * x_scalar)

    def _normalize(self, x_scalar: jnp.ndarray) -> jnp.ndarray:
        var = jnp.maximum(self.ema_sq - self.ema_mean * self.ema_mean, self.eps)
        std = jnp.sqrt(var)
        return (x_scalar - self.ema_mean) / std

    def compute(self, obs) -> float:
        x = jnp.asarray(obs, jnp.float32)
        _, se = _loss_and_se(self.pred, self.tgt, x)
        raw = jnp.mean(se)
        norm = self._normalize(raw)
        return float(norm)

    def update(self, obs) -> float:
        x = jnp.asarray(obs, jnp.float32)

        def loss_fn(p):
            l, _ = _loss_and_se(p, self.tgt, x)
            return l

        loss, grads = jax.value_and_grad(loss_fn)(self.pred)
        updates, self.opt_state = self.opt.update(grads, self.opt_state, params=self.pred)
        self.pred = optax.apply_updates(self.pred, updates)
        return float(loss)

    def compute_and_update(self, obs) -> float:
        x = jnp.asarray(obs, jnp.float32)
        loss, se = _loss_and_se(self.pred, self.tgt, x)
        raw = jnp.mean(se)
        self._update_norm(raw)
        norm = self._normalize(raw)

        grads = jax.grad(lambda p: _loss_and_se(p, self.tgt, x)[0])(self.pred)
        updates, self.opt_state = self.opt.update(grads, self.opt_state, params=self.pred)
        self.pred = optax.apply_updates(self.pred, updates)
        return float(norm)
