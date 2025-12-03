# generar_policy.py
# Genera un checkpoint Cap'n Proto (Float64 / UInt64) para la policy,
# con acId=xxh3_64(phi) y Pa opcional si es compatible — solo JAX (sin NumPy)
# CLI minimalista sin argparse/typer (baja sobrecarga).

import os, sys, struct
import capnp  # type: ignore

# Habilitar Float64 en JAX ANTES de importarlo (evita truncamientos)
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import xxhash  # type: ignore
from typing import Tuple, Dict

from config import BOARD_SIZE
import edificios, areas
from ml_policy import Policy

# ------- Esquema -------
policy_schema = capnp.load("policy.capnp")  # type: ignore
SchemaPolicy = policy_schema.PolicyCheckpoint

# ------- Defs del entorno (idéntico a modo_automatico) -------
S = int(BOARD_SIZE)
ED = edificios.edificios
ACTION_SET = (
    'residencia','taller_togas','herreria','refineria','lecheria',
    'agua','depuradora','decoracion','policia','bombero',
    'colegio','hospital','suelo','demoler'
)
SERVICIOS = ('policia','bombero','colegio','hospital')

# ------- CLI minimal -------
def _print_help(prog: str):
    print(f"""Uso: {prog} [--out RUTA] [--seed N] [--hidden N] [--help]
  --out RUTA     Ruta de salida del .bin (default: ml_data/policy/latest.ckpt.bin)
  --seed N       Semilla de inicialización (default: 42)
  --hidden N     Tamaño de la capa oculta (default: 128)
  --help         Muestra esta ayuda
""")

def parse_cli(argv) -> Dict[str, str]:
    # Soporta --key=val o --key val
    opts: Dict[str, str] = {"out": "ml_data/policy/latest.ckpt.bin", "seed": "42", "hidden": "128"}
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in ("-h", "--help"):
            _print_help(sys.argv[0]); sys.exit(0)
        if tok.startswith("--"):
            if "=" in tok:
                k, v = tok[2:].split("=", 1)
                if k in ("out", "seed", "hidden"): opts[k] = v
            else:
                k = tok[2:]
                if k in ("out", "seed", "hidden") and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                    opts[k] = argv[i + 1]; i += 1
        i += 1
    return opts

# ------- φ(a): contexto por acción (Float64) -------
def _absmax(vals):
    m = 1.0
    for v in vals:
        av = abs(float(v))
        if av > m: m = av
    return m

def _build_phi() -> jnp.ndarray:
    energia    = [ED[b].energia    for b in ED if b in ED]
    agua       = [ED[b].agua       for b in ED if b in ED]
    comida     = [ED[b].comida     for b in ED if b in ED]
    basura     = [ED[b].basura     for b in ED if b in ED]
    empleos    = [ED[b].empleos    for b in ED if b in ED]
    residentes = [ED[b].residentes for b in ED if b in ED]
    felicidad  = [ED[b].felicidad  for b in ED if b in ED]
    ambiente   = [ED[b].ambiente   for b in ED if b in ED]
    ws = [ED[b].tamanio[0] for b in ED if b in ED]
    hs = [ED[b].tamanio[1] for b in ED if b in ED]

    mE  = _absmax(energia);    mA = _absmax(agua)
    mC  = _absmax(comida);     mB = _absmax(basura)
    mJ  = _absmax(empleos);    mR = _absmax(residentes)
    mF  = _absmax(felicidad);  mAM = _absmax(ambiente)
    mW  = max(1.0, max(ws) if ws else 1.0)
    mH  = max(1.0, max(hs) if hs else 1.0)

    TIPOS   = ('residencia','comercio','industria','decoracion','suelo')
    PROVIDE = ('energia','agua','comida','basura')
    SERV    = SERVICIOS

    cx = cy = S // 2
    def zona_size(bn: str) -> float:
        try:
            return float(len(areas.Area.zona_cubierta_por_edificio(bn, (cx, cy), S)))
        except Exception:
            return 0.0
    denom_area = float(S * S) if S > 0 else 1.0

    rows: list[list[float]] = []
    for a in ACTION_SET:
        if a not in ED:  # 'demoler', etc.
            rows.append([0.0] * (8 + 2 + len(TIPOS) + len(PROVIDE) + len(SERV) + 3))
            continue
        e = ED[a]
        w, h = e.tamanio
        base = [
            e.energia/mE, e.agua/mA, e.comida/mC, e.basura/mB,
            e.empleos/mJ, e.residentes/mR, e.felicidad/mF, e.ambiente/mAM,
            float(w)/mW, float(h)/mH
        ]
        tname = str(getattr(e.tipo, 'value', e.tipo)).lower()
        tvec  = [1.0 if tname == t else 0.0 for t in TIPOS]
        pvec  = [1.0 if getattr(e, PROVIDE[i]) > 0 else 0.0 for i in range(len(PROVIDE))]
        svec  = [1.0 if a == srv else 0.0 for srv in SERV]
        area  = zona_size(a) / denom_area
        area_fel = area if e.felicidad > 0 else 0.0
        area_eco = area if e.ambiente  > 0 else 0.0
        area_srv = area if a in SERVICIOS else 0.0
        rows.append(base + tvec + pvec + svec + [area_fel, area_eco, area_srv])

    return jnp.asarray(rows, dtype=jnp.float64)

# ------- acId: xxh3_64(phi) sin NumPy -------
def ac_id_from_phi(phi: jnp.ndarray) -> int:
    phi = jnp.asarray(phi, dtype=jnp.float64)
    rows, cols = int(phi.shape[0]), int(phi.shape[1])
    h = xxhash.xxh3_64()
    # tamaños en UInt64 LE
    h.update(struct.pack("<Q", rows))
    h.update(struct.pack("<Q", cols))
    # datos row-major en Float64
    flat = jnp.ravel(phi)
    host = jax.device_get(flat)         # ndarray host (NumPy interno de JAX; no importamos numpy)
    h.update(host.tobytes(order="C"))   # bytes en orden C
    return h.intdigest()

# ------- helpers: listas Float64 para Cap'n Proto -------
def _f64_list(x) -> list[float]:
    arr = jnp.asarray(x, jnp.float64)
    return [float(v) for v in jnp.ravel(arr)]

def _shape2(x) -> Tuple[int, int]:
    s = tuple(int(d) for d in jnp.shape(x))
    assert len(s) == 2
    return s[0], s[1]

# ------- volcado a Cap'n Proto -------
def save_policy_capnp(policy: Policy, phi: jnp.ndarray, out_path: str, version: int = 1):
    dirp = os.path.dirname(out_path)
    if dirp:
        os.makedirs(dirp, exist_ok=True)

    # Dimensiones
    W1 = policy.params['W1']; b1 = policy.params['b1']
    Wt = policy.params['Wt']; bt = policy.params['bt']
    Wp = policy.params['Wp']; bp = policy.params['bp']

    din, dh = _shape2(W1)
    dh2, na = _shape2(Wt)
    assert dh2 == dh, "Wt.shape[0] debe igualar hidden"
    _, hp = _shape2(Wp)
    assert hp == 4, "Wp debe tener 4 columnas (head de coords)"

    phi = jnp.asarray(phi, dtype=jnp.float64)
    phiRows, phiCols = int(phi.shape[0]), int(phi.shape[1])
    acId = ac_id_from_phi(phi)

    # Mensaje
    msg = SchemaPolicy.new_message()
    msg.version     = int(version)
    msg.obsDim      = int(din)
    msg.hidden      = int(dh)
    msg.actionCount = int(na)

    msg.phiRows = int(phiRows)
    msg.phiCols = int(phiCols)
    msg.acId    = int(acId)

    # Tensores (Float64 row-major como listas)
    # ⚠️ Campos en minúscula (coinciden con policy.capnp)
    tW1 = msg.w1
    tW1.rows = int(din); tW1.cols = int(dh); tW1.data = _f64_list(W1)

    tb1 = msg.b1
    tb1.data = _f64_list(b1)

    tWt = msg.wt
    tWt.rows = int(dh); tWt.cols = int(na); tWt.data = _f64_list(Wt)

    tbt = msg.bt
    tbt.data = _f64_list(bt)

    tWp = msg.wp
    tWp.rows = int(dh); tWp.cols = int(4); tWp.data = _f64_list(Wp)

    tbp = msg.bp
    tbp.data = _f64_list(bp)

    # Pa opcional, solo si es compatible (pa en minúscula)
    hasPa = ('Pa' in policy.params)
    if hasPa:
        Pa = policy.params['Pa']
        k, dh_pa = _shape2(Pa)
        if (k == phiCols) and (dh_pa == dh):
            msg.hasPa = True
            tPa = msg.pa
            tPa.rows = int(k); tPa.cols = int(dh); tPa.data = _f64_list(Pa)
        else:
            msg.hasPa = False
    else:
        msg.hasPa = False

    # Escritura atómica
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        msg.write(f)
    os.replace(tmp, out_path)
    print(f"[OK] Guardado checkpoint Cap'n Proto → {out_path}")

# ------- main -------
def main():
    opts = parse_cli(sys.argv[1:])
    out_path = opts["out"]
    seed = int(opts["seed"])
    hidden = int(opts["hidden"])

    # obsDim = 11 (stats) + |ACTION_SET| + 4 (coberturas)
    obs_dim = 11 + len(ACTION_SET) + 4

    # φ(a)
    phi = _build_phi()

    # Crear policy nueva y asegurar Pa compatible con φ(a)
    policy = Policy(action_count=len(ACTION_SET), obs_dim=obs_dim, seed=seed, hidden=hidden, entropy_coef=0.007)
    try:
        policy.action_context = jnp.asarray(phi, dtype=jnp.float32)
        _ = policy._ensure_ctx()  # type: ignore
    except Exception:
        pass

    save_policy_capnp(policy, phi, out_path, version=1)

if __name__ == "__main__":
    main()
