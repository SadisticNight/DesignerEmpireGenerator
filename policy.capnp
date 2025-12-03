@0xabcdeffedcba9876;

struct Tensor1F64 {
  data @0 :List(Float64);         # vector de 64-bit floats
}

struct Tensor2F64 {
  rows @0 :UInt64;                # dimensión 0 (filas)
  cols @1 :UInt64;                # dimensión 1 (columnas)
  data @2 :List(Float64);         # row-major (rows*cols)
}

struct PolicyCheckpoint {
  version      @0 :UInt64;        # versión del ckpt
  obsDim       @1 :UInt64;        # dimensión de la observación
  hidden       @2 :UInt64;        # tamaño de la capa oculta
  actionCount  @3 :UInt64;        # cantidad de acciones

  # Identidad del contexto de acción φ(a)
  phiRows      @4 :UInt64;        # normalmente = actionCount
  phiCols      @5 :UInt64;        # K de φ(a)
  acId         @6 :UInt64;        # xxh3_64(phiRows, phiCols, bytes(Float64) row-major)

  # Pesos principales (Float64) — nombres en minúscula
  w1 @7  :Tensor2F64;
  b1 @8  :Tensor1F64;
  wt @9  :Tensor2F64;
  bt @10 :Tensor1F64;
  wp @11 :Tensor2F64;
  bp @12 :Tensor1F64;

  # Cabezal contextual (opcional) — válido solo si hasPa y compatible
  hasPa @13 :Bool;
  pa    @14 :Tensor2F64;          # (phiCols, hidden)
}
