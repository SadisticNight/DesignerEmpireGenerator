# ml_replay.py — Writer/Reader Cap’n Proto + export Parquet + dataset JAX (+ sync)
from __future__ import annotations

import os, time, tempfile
from typing import Iterator, List
import jax, jax.numpy as jnp

# --- deps opcionales ---
try:
    import capnp  # pycapnp
except Exception as e:
    capnp = None
    _CAPNP_ERR = e

try:
    import polars as pl
except Exception:
    pl = None

# --- esquema inline (evita depender del CWD) ---
_SCHEMA = r'''
@0xdecafbadc0ffeeaa;
struct Step {
  version    @0 :UInt16;
  episodeId  @1 :UInt64;
  t          @2 :UInt32;
  ok         @3 :UInt8;
  actionType @4 :UInt16;
  u          @5 :Float32;
  v          @6 :Float32;
  reward     @7 :Float32;
  ecProxy    @8 :Float32;
  obs        @9 :List(Float32);
}
'''

def _load_schema():
    if capnp is None:
        raise RuntimeError(
            f"capnp requerido (pip install pycapnp). Error base: {_CAPNP_ERR!r}"
            if '_CAPNP_ERR' in globals() else "capnp requerido"
        )
    tmp = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.capnp')
    try:
        tmp.write(_SCHEMA)
        tmp.flush()
        m = capnp.load(tmp.name)
    finally:
        try: tmp.close()
        except Exception: pass
        try: os.remove(tmp.name)
        except Exception: pass
    return m

_SCH = _load_schema() if capnp else None

# =========================
# Writer (append-only, packed)
# =========================
class StepWriter:
    __slots__ = ('episode_id','path','f','buf','t','version','flush_every')

    def __init__(self, base_dir: str = 'ml_data/episodes', episode_id: int | None = None,
                 version: int = 1, flush_every: int = 1024):
        if _SCH is None:
            raise RuntimeError("capnp no disponible: no se pudo cargar el esquema Step.")
        os.makedirs(base_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        self.episode_id = int(episode_id if episode_id is not None else ts)
        self.path = os.path.join(base_dir, f'ep_{self.episode_id}.capnp.bin')
        self.f = open(self.path, 'ab', buffering=0)  # sin buffer Python
        self.buf: List[bytes] = []
        self.t = 0
        self.version = int(version)
        self.flush_every = int(flush_every)

    def append(self, ok: int, action_type: int, u: float, v: float, reward: float, ec_proxy: float, obs_arr):
        msg = _SCH.Step.new_message()
        msg.version    = self.version
        msg.episodeId  = self.episode_id
        msg.t          = self.t
        msg.ok         = int(ok)
        msg.actionType = int(action_type)
        msg.u          = float(u)
        msg.v          = float(v)
        msg.reward     = float(reward)
        msg.ecProxy    = float(ec_proxy)

        obs_flat = jnp.ravel(jnp.asarray(obs_arr, jnp.float32)).tolist()
        arr = msg.init('obs', len(obs_flat))
        arr[:] = obs_flat

        self.buf.append(msg.to_bytes_packed())
        self.t += 1
        if len(self.buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if self.buf:
            self.f.write(b''.join(self.buf))
            self.buf.clear()

    def sync(self):
        try:
            self.flush()
            self.f.flush()
            os.fsync(self.f.fileno())
        except Exception:
            pass

    def close(self):
        try:
            self.flush()
        finally:
            try: self.f.close()
            except Exception: pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

# =========================
# Reader (streaming, packed; tolera final truncado)
# =========================
def iter_steps_file(path: str) -> Iterator:
    """Lee una secuencia de Step *packed* desde 'path'. Ignora un último mensaje
    parcialmente escrito (p.ej., si el writer estaba en curso) en lugar de fallar."""
    if _SCH is None or capnp is None:
        raise RuntimeError("capnp no disponible: no se pudo cargar el esquema Step.")

    with open(path, 'rb') as f:
        while True:
            pos = f.tell()
            try:
                st = _SCH.Step.read_packed(f)
                yield st
            except Exception as ex:
                # pycapnp puede lanzar EOFError o KjException("Premature EOF")
                try:
                    end = f.seek(0, os.SEEK_END)
                    end = f.tell()
                except Exception:
                    end = pos  # si falla, asumimos EOF
                # Si estábamos exactamente al final → EOF limpio
                if pos == end:
                    break
                # Si restan menos de 8 bytes (menor a header mínimo) → tail truncado: se ignora
                if (end - pos) < 8:
                    break
                # Si es KjException con 'Premature EOF', asumimos tail truncado y salimos
                if ex.__class__.__name__ == "KjException" and "Premature EOF" in str(ex):
                    break
                # Cualquier otro caso es error real
                raise

def iter_steps_dir(capnp_dir: str = 'ml_data/episodes', limit_files: int | None = None) -> Iterator:
    files = [os.path.join(capnp_dir, x) for x in os.listdir(capnp_dir) if x.endswith('.capnp.bin')]
    files.sort()
    if limit_files:
        files = files[:int(limit_files)]
    for p in files:
        for st in iter_steps_file(p):
            yield st

# =========================
# Export (opcional con Polars)
# =========================
def export_parquet(capnp_dir: str = 'ml_data/episodes',
                   out_parquet: str = 'ml_data/episodes.parquet',
                   limit_files: int = 64) -> bool:
    if pl is None:
        raise RuntimeError("polars no disponible: pip install polars")
    rows = []
    for st in iter_steps_dir(capnp_dir, limit_files):
        rows.append({
            'episodeId': int(st.episodeId),
            't':         int(st.t),
            'ok':        int(st.ok),
            'actionType':int(st.actionType),
            'u':         float(st.u),
            'v':         float(st.v),
            'reward':    float(st.reward),
            'ecProxy':   float(st.ecProxy),
        })
    if not rows:
        return False
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_parquet, compression='zstd')
    return True

# =========================
# Dataset JAX (para training)
# =========================
def load_dataset(capnp_dir: str = 'ml_data/episodes',
                 max_steps: int = 1_000_000,
                 shuffle: bool = True,
                 seed: int = 0):
    """Devuelve tensores JAX: (obs, act, u, v, rw, ec, epi, tt)"""
    if _SCH is None or capnp is None:
        raise RuntimeError("capnp no disponible: asegurate de tener pycapnp instalado.")

    obs: List[List[float]] = []
    act: List[int] = []
    u:   List[float] = []
    v:   List[float] = []
    rw:  List[float] = []
    ec:  List[float] = []
    epi: List[int] = []
    tt:  List[int] = []

    c = 0
    for st in iter_steps_dir(capnp_dir):
        obs.append(list(st.obs))
        act.append(int(st.actionType))
        u.append(float(st.u))
        v.append(float(st.v))
        rw.append(float(st.reward))
        ec.append(float(st.ecProxy))
        epi.append(int(st.episodeId))
        tt.append(int(st.t))
        c += 1
        if c >= int(max_steps):
            break

    if not obs:
        raise RuntimeError("sin datos: no se encontraron pasos en el directorio especificado")

    obs = jnp.asarray(obs, jnp.float32)
    act = jnp.asarray(act, jnp.int32)
    u   = jnp.asarray(u,   jnp.float32)
    v   = jnp.asarray(v,   jnp.float32)
    rw  = jnp.asarray(rw,  jnp.float32)
    ec  = jnp.asarray(ec,  jnp.float32)
    epi = jnp.asarray(epi, jnp.int64)
    tt  = jnp.asarray(tt,  jnp.int32)

    if shuffle:
        key = jax.random.PRNGKey(int(seed))
        idx = jax.random.permutation(key, obs.shape[0])
        obs, act, u, v, rw, ec, epi, tt = obs[idx], act[idx], u[idx], v[idx], rw[idx], ec[idx], epi[idx], tt[idx]

    return obs, act, u, v, rw, ec, epi, tt
