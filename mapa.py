# save_map_png_mgl_jax.py
# Requisitos: pip install moderngl glcontext pillow pycapnp jax
# (sin numpy; usamos jax.numpy y device_get para volcar buffers)

import sys
import jax
import jax.numpy as jnp
import moderngl as mgl
from PIL import Image, ImageOps
import capnp

CC = capnp.load("celdas.capnp")
EC = capnp.load("edificios.capnp")

def load_map(celdas_bin="celdas.bin", edificios_bin="edificios.bin"):
    with open(celdas_bin, "rb") as f:
        mapa = CC.Mapa.read(f)
    with open(edificios_bin, "rb") as f:
        eds  = EC.Edificios.read(f)

    pal = {b.nombre: tuple(b.color) for b in eds.lista}  # nombre -> (R,G,B)

    # recolectar posiciones y colores (u8) desde Cap'n Proto
    pos_list = []
    col_u8_list = []
    maxx = max((int(c.x) for c in mapa.celdas), default=0)
    maxy = max((int(c.y) for c in mapa.celdas), default=0)
    for c in mapa.celdas:
        if c.hash and (c.edificio in pal):
            pos_list.append((float(c.x), float(c.y)))
            r, g, b = pal[c.edificio]
            col_u8_list.append((float(r), float(g), float(b)))

    if not pos_list:
        pos_list = [(0.0, 0.0)]
        col_u8_list = [(255.0, 255.0, 255.0)]

    # a JAX (f32)
    pos_f32 = jnp.asarray(pos_list, dtype=jnp.float32)
    col_u8_f32 = jnp.asarray(col_u8_list, dtype=jnp.float32)

    pos_f32, col_f32 = _normalize_colors(pos_f32, col_u8_f32)
    W, H = maxx + 1, maxy + 1
    return pos_f32, col_f32, (W, H)

@jax.jit
def _normalize_colors(pos_f32, col_u8_f32):
    # normaliza [0..255] -> [0..1] en JIT
    return pos_f32, col_u8_f32 / jnp.float32(255.0)

@jax.jit
def _unit_quad():
    # dos triángulos para un quad 1x1 centrado
    return jnp.array([
        [-0.5, -0.5], [ 0.5, -0.5], [ 0.5,  0.5],
        [-0.5, -0.5], [ 0.5,  0.5], [-0.5,  0.5],
    ], dtype=jnp.float32)

def _to_bytes_f32(arr):
    # sin importar numpy: usamos device_get → ndarray host y .tobytes()
    host = jax.device_get(jnp.asarray(arr, jnp.float32))
    return host.tobytes(order="C")

VERT = """
#version 330
in vec2 in_pos;      // quad unitario [-0.5..0.5]
in vec2 i_xy;        // posición de celda (en celdas)
in vec3 i_col;       // color
uniform vec2 u_map_px;   // tamaño del framebuffer (px)
uniform float u_cell_px; // px por celda
out vec3 v_col;
void main() {
    vec2 px = (i_xy * u_cell_px) + (in_pos * u_cell_px);
    vec2 ndc = (px / u_map_px) * 2.0 - 1.0;
    ndc.y = -ndc.y; // origen arriba-izquierda
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_col = i_col;
}
""";

FRAG = """
#version 330
in vec3 v_col;
out vec4 f_col;
void main() { f_col = vec4(v_col, 1.0); }
""";

def render_png(out_path="mapa.png", cell_px=12, celdas_bin="celdas.bin", edificios_bin="edificios.bin"):
    pos, col, (W, H) = load_map(celdas_bin, edificios_bin)
    width, height = int(W * cell_px), int(H * cell_px)

    ctx = mgl.create_standalone_context()
    fbo = ctx.simple_framebuffer((width, height))
    fbo.use()
    fbo.clear(1.0, 1.0, 1.0, 1.0)

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    quad = _unit_quad()

    vbo      = ctx.buffer(_to_bytes_f32(quad))
    inst_xy  = ctx.buffer(_to_bytes_f32(pos))
    inst_col = ctx.buffer(_to_bytes_f32(col))

    vao = ctx.vertex_array(
        prog,
        [(vbo,      "2f",   "in_pos"),
         (inst_xy,  "2f/i", "i_xy"),
         (inst_col, "3f/i", "i_col")]
    )

    prog["u_map_px"].value = (float(width), float(height))
    prog["u_cell_px"].value = float(cell_px)

    vao.render(mode=mgl.TRIANGLES, instances=int(pos.shape[0]))

    data = fbo.read(components=3, alignment=1)
    img = Image.frombytes("RGB", (width, height), data)
    img = ImageOps.flip(img)
    img.save(out_path, "PNG")
    print(f"[OK] PNG guardado → {out_path}  ({width}x{height})")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "mapa.png"
    px  = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    render_png(out_path=out, cell_px=px)
