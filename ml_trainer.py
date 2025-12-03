# ml_trainer.py — BC / PG offline con JAX puro (sin Distrax)
# - Carga dataset Cap’n Proto vía ml_replay
# - Entrena Policy en modo BC (clonado) o PG (REINFORCE) offline
# - Guarda a .pkl (por default) compatible con Policy.load()

from __future__ import annotations
import os, sys
import jax, jax.numpy as jnp

from ml_policy import Policy
from ml_replay import load_dataset


# ------- CLI minimal (sin argparse) -------
def _print_help(prog: str):
    print(f"""Uso: {prog} [--mode bc|pg] [--data DIR] [--out RUTA.pkl]
             [--epochs N] [--batch N] [--lr F] [--coord_weight F] [--seed N] [--help]

  --mode           bc | pg              (default: bc)
  --data           directorio dataset   (default: ml_data/episodes)
  --out            ruta ckpt .pkl/.bin  (default: ml_data/policy/latest.ckpt.pkl)
  --epochs         épocas de entrenamiento (default: 5)
  --batch          tamaño de batch         (default: 2048 para BC, 4096 para PG)
  --lr             learning rate           (default: 3e-4)
  --coord_weight   peso NLL coords (solo BC) (default: 5.0)
  --seed           semilla                 (default: 42)
  --help           muestra esta ayuda

Ejemplos:
  BC (clonado):
    python ml_trainer.py --mode bc --data ml_data/episodes \\
      --out ml_data/policy/latest.ckpt.pkl --epochs 10 --batch 4096 --lr 3e-4 --coord_weight 5.0 --seed 42

  PG offline (REINFORCE sobre replays):
    python ml_trainer.py --mode pg --data ml_data/episodes \\
      --out ml_data/policy/latest.ckpt.pkl --epochs 10 --batch 8192 --lr 3e-4 --seed 42
""")


def _parse_cli(argv):
    # batch=None para poder elegir default según --mode en runtime
    opts = {
        "mode": "bc",
        "data": "ml_data/episodes",
        "out":  "ml_data/policy/latest.ckpt.pkl",
        "epochs": "5",
        "batch": None,
        "lr": "3e-4",
        "coord_weight": "5.0",
        "seed": "42",
    }
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-h", "--help"):
            _print_help(sys.argv[0]); sys.exit(0)
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok[2:].split("=", 1)
            else:
                k = tok[2:]; v = None
                if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    v = argv[i + 1]; i += 1
            if k in opts:
                opts[k] = v if v is not None else "1"
        i += 1
    return opts


# ------- utils -------
def _batches(N, bs):
    i = 0
    while i < N:
        j = i + bs if i + bs < N else N
        yield slice(i, j); i = j


@jax.jit
def _compute_returns(epi, tt, rw, gamma=0.98):
    """
    Calcula RTG por episodio respetando discontinuidades en (episodeId, t).
    """
    idx = jnp.lexsort((tt, epi))          # orden por epi y luego por t
    e = epi[idx]; t = tt[idx]; r = rw[idx]

    def scan_fn(carry, inp):
        last_e, last_t, G = carry
        ee, ttt, rr = inp
        reset = (ee != last_e) | (ttt != last_t - 1)
        G = jnp.where(reset, 0.0, G * gamma) + rr
        return (ee, ttt, G), G

    (_, _, _), Gseq = jax.lax.scan(
        scan_fn,
        (jnp.int64(-1), jnp.int32(-1), jnp.float32(0.0)),
        (e, t, r),
        reverse=True
    )
    rtg = jnp.zeros_like(r).at[idx].set(Gseq)
    # normalizar ventajas
    m = jnp.mean(rtg); s = jnp.std(rtg) + 1e-6
    return (rtg - m) / s


# ------- entrenamiento -------
def train_bc(data_dir='ml_data/episodes', out_path='ml_data/policy/latest.ckpt.pkl',
             epochs=5, batch_size=2048, lr=3e-4, coord_weight=5.0, seed=42):
    obs, act, u, v, rw, ec, epi, tt = load_dataset(data_dir, max_steps=2_000_000, shuffle=True, seed=seed)
    N = int(obs.shape[0])
    if N == 0:
        raise RuntimeError(f"No hay pasos en '{data_dir}'.")
    pol = Policy(action_count=int(jnp.max(act)) + 1, obs_dim=int(obs.shape[1]),
                 seed=seed, hidden=128, entropy_coef=0.0)
    # si existe, intenta continuar desde un ckpt previo
    try: pol.load(out_path)
    except Exception: pass

    print(f"[BC] dataset={N} pasos | obs_dim={int(obs.shape[1])} | actions={int(jnp.max(act))+1} | batch={batch_size}")
    for ep in range(1, epochs + 1):
        key = jax.random.PRNGKey(seed + ep)
        idx = jax.random.permutation(key, N)
        o, ac, uu, vv = obs[idx], act[idx], u[idx], v[idx]
        losses = []
        for sl in _batches(N, int(batch_size)):
            loss = pol.bc_update(o[sl], ac[sl], uu[sl], vv[sl], lr=lr, coord_weight=coord_weight)
            losses.append(float(loss))
        print(f"[epoch {ep}] loss={sum(losses)/max(1,len(losses)):.6f} (BC)")
    pol.save(out_path)
    return out_path


def train_pg_offline(data_dir='ml_data/episodes', out_path='ml_data/policy/latest.ckpt.pkl',
                     epochs=5, batch_size=4096, lr=3e-4, seed=42):
    obs, act, u, v, rw, ec, epi, tt = load_dataset(data_dir, max_steps=2_000_000, shuffle=True, seed=seed)
    N = int(obs.shape[0])
    if N == 0:
        raise RuntimeError(f"No hay pasos en '{data_dir}'.")
    adv = _compute_returns(epi, tt, rw, gamma=0.98)

    pol = Policy(action_count=int(jnp.max(act)) + 1, obs_dim=int(obs.shape[1]),
                 seed=seed, hidden=128, entropy_coef=0.001)
    try: pol.load(out_path)
    except Exception: pass

    print(f"[PG] dataset={N} pasos | obs_dim={int(obs.shape[1])} | actions={int(jnp.max(act))+1} | batch={batch_size}")
    for ep in range(1, epochs + 1):
        key = jax.random.PRNGKey(seed + 100 + ep)
        idx = jax.random.permutation(key, N)
        o, ac, uu, vv, aa = obs[idx], act[idx], u[idx], v[idx], adv[idx]
        losses = []
        for sl in _batches(N, int(batch_size)):
            loss = pol.update(o[sl], ac[sl], uu[sl], vv[sl], aa[sl], lr=lr)
            losses.append(float(loss))
        print(f"[epoch {ep}] loss={sum(losses)/max(1,len(losses)):.6f} (PG-offline)")
    pol.save(out_path)
    return out_path


# ------- main -------
def main():
    o = _parse_cli(sys.argv[1:])
    mode = (o["mode"] or "bc").lower()
    data = o["data"] or "ml_data/episodes"
    out  = o["out"]  or "ml_data/policy/latest.ckpt.pkl"
    epochs = int(o["epochs"] or "5")
    # default dinámico de batch según modo si el usuario no lo pasó
    batch_default = 4096 if mode == "pg" else 2048
    batch = int(o["batch"]) if (o["batch"] is not None) else batch_default
    lr = float(o["lr"] or "3e-4")
    coord_weight = float(o["coord_weight"] or "5.0")
    seed = int(o["seed"] or "42")

    if mode == "bc":
        train_bc(data_dir=data, out_path=out, epochs=epochs, batch_size=batch,
                 lr=lr, coord_weight=coord_weight, seed=seed)
    elif mode == "pg":
        train_pg_offline(data_dir=data, out_path=out, epochs=epochs, batch_size=batch,
                         lr=lr, seed=seed)
    else:
        _print_help(sys.argv[0]); sys.exit(2)


if __name__ == '__main__':
    main()
