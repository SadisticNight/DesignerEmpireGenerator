@0xabcdefabcdefabcd;

# Este struct representa la cantidad de un edificio.
struct CantidadEdificio {
  nombre @0 :Text;
  cantidad @1 :Int64;
}

# Este struct define las estadisticas generales.
struct Stats {
  totalResidentes @0 :Int64;
  totalEmpleosIndustria @1 :Int64;
  totalEmpleosComercio @2 :Int64;
  totalEmpleos @3 :Int64;
  porcentajeIndustria @4 :Float64;
  porcentajeComercio @5 :Float64;
  desequilibrioLaboral @6 :Int64;
  ec @7 :Bool;
  energiaUsada @8 :Int64;
  energiaTotal @9 :Int64;
  aguaUsada @10 :Int64;
  aguaTotal @11 :Int64;
  comidaUsada @12 :Int64;
  comidaTotal @13 :Int64;
  basuraUsada @14 :Int64;
  basuraTotal @15 :Int64;
  ecologiaTotal @16 :Int64;
  felicidadTotal @17 :Int64;
  cantidadEdificios @18 :List(CantidadEdificio);
}
