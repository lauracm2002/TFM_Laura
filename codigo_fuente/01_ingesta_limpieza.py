from pathlib import Path
import pandas as pd

# =========================
# CONFIGURACIÓN GENERAL
# =========================

BASE = Path(".")
RUTA_HISTORICO = BASE / "datos" / "originales" / "historico_trafico"
RUTA_PUNTOS = BASE / "datos" / "originales" / "puntos_medida"
RUTA_PROCESADOS = BASE / "datos" / "procesados"

RUTA_PROCESADOS.mkdir(parents=True, exist_ok=True)

COLUMNAS_HISTORICO = [
    "id",
    "fecha",
    "tipo_elem",
    "intensidad",
    "ocupacion",
    "carga",
    "vmed",
    "error",
    "periodo_integracion"
]

CHUNKSIZE = 500_000
MAX_PUNTOS = 50


# =========================
# FUNCIONES AUXILIARES
# =========================

def leer_puntos_medida():
    """
    Lee el archivo Excel de puntos de medida y devuelve una tabla limpia
    con id, nombre, distrito, longitud y latitud.
    """
    archivos = list(RUTA_PUNTOS.glob("*.xlsx")) + list(RUTA_PUNTOS.glob("*.xls"))

    if not archivos:
        raise FileNotFoundError("No se ha encontrado ningún archivo Excel en datos/originales/puntos_medida")

    archivo = archivos[0]
    print(f"Leyendo puntos de medida: {archivo.name}")

    puntos = pd.read_excel(archivo)

    puntos.columns = (
        puntos.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    columnas_necesarias = ["id", "tipo_elem", "distrito", "nombre", "longitud", "latitud"]
    columnas_existentes = [c for c in columnas_necesarias if c in puntos.columns]

    puntos = puntos[columnas_existentes].copy()
    puntos = puntos.drop_duplicates(subset=["id"])

    puntos["id"] = pd.to_numeric(puntos["id"], errors="coerce")
    puntos = puntos.dropna(subset=["id"])
    puntos["id"] = puntos["id"].astype(int)

    if "longitud" in puntos.columns:
        puntos["longitud"] = pd.to_numeric(puntos["longitud"], errors="coerce")

    if "latitud" in puntos.columns:
        puntos["latitud"] = pd.to_numeric(puntos["latitud"], errors="coerce")

    puntos.to_csv(RUTA_PROCESADOS / "puntos_medida_limpios.csv", index=False, sep=";")

    print(f"Puntos de medida limpios: {len(puntos)}")
    return puntos


def seleccionar_top_puntos(archivo_referencia):
    """
    Para que el proceso sea rápido y viable para la entrega,
    selecciona los puntos con más registros válidos en el primer archivo.
    """
    print(f"Seleccionando los {MAX_PUNTOS} puntos con más registros usando: {archivo_referencia.name}")

    conteos = {}

    for chunk in pd.read_csv(
        archivo_referencia,
        sep=";",
        encoding="utf-8",
        usecols=["id", "error"],
        chunksize=CHUNKSIZE
    ):
        chunk = chunk[chunk["error"] == "N"]
        vc = chunk["id"].value_counts()

        for id_punto, n in vc.items():
            conteos[id_punto] = conteos.get(id_punto, 0) + n

    serie = pd.Series(conteos).sort_values(ascending=False)
    top_ids = serie.head(MAX_PUNTOS).index.astype(int).tolist()

    print(f"Puntos seleccionados: {len(top_ids)}")
    print(f"Primeros puntos seleccionados: {top_ids[:10]}")

    pd.DataFrame({
        "id": top_ids,
        "registros_archivo_referencia": serie.head(MAX_PUNTOS).values
    }).to_csv(RUTA_PROCESADOS / "puntos_seleccionados_modelo.csv", index=False, sep=";")

    return top_ids


def procesar_archivo_historico(archivo, ids_seleccionados):
    """
    Procesa un CSV mensual por bloques:
    - filtra errores
    - convierte fechas
    - convierte numéricos
    - filtra intensidades negativas
    - agrega por hora y punto de medida
    """
    print(f"\nProcesando archivo: {archivo.name}")

    grupos_archivo = []
    registros_leidos = 0
    registros_validos = 0

    for chunk in pd.read_csv(
        archivo,
        sep=";",
        encoding="utf-8",
        usecols=COLUMNAS_HISTORICO,
        chunksize=CHUNKSIZE
    ):
        registros_leidos += len(chunk)

        # Seleccionar solo puntos elegidos
        chunk = chunk[chunk["id"].isin(ids_seleccionados)]

        # Filtrar registros sin error
        chunk = chunk[chunk["error"] == "N"]

        # Convertir fecha
        chunk["fecha"] = pd.to_datetime(chunk["fecha"], errors="coerce")
        chunk = chunk.dropna(subset=["fecha"])

        # Convertir numéricos
        for col in ["intensidad", "ocupacion", "carga", "vmed", "periodo_integracion"]:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # Eliminar intensidad nula o negativa
        chunk = chunk.dropna(subset=["intensidad"])
        chunk = chunk[chunk["intensidad"] >= 0]

        registros_validos += len(chunk)

        # Agregación horaria
        chunk["fecha_hora"] = chunk["fecha"].dt.floor("h")

        grupo = (
            chunk
            .groupby(["id", "fecha_hora"], as_index=False)
            .agg(
                intensidad_horaria=("intensidad", "sum"),
                ocupacion_media=("ocupacion", "mean"),
                carga_media=("carga", "mean"),
                velocidad_media=("vmed", "mean"),
                periodo_integracion_medio=("periodo_integracion", "mean"),
                registros_15min=("intensidad", "count")
            )
        )

        grupos_archivo.append(grupo)

    if grupos_archivo:
        resultado = pd.concat(grupos_archivo, ignore_index=True)

        # Reagrupar por si el mismo id/hora aparece en distintos chunks
        resultado = (
            resultado
            .groupby(["id", "fecha_hora"], as_index=False)
            .agg(
                intensidad_horaria=("intensidad_horaria", "sum"),
                ocupacion_media=("ocupacion_media", "mean"),
                carga_media=("carga_media", "mean"),
                velocidad_media=("velocidad_media", "mean"),
                periodo_integracion_medio=("periodo_integracion_medio", "mean"),
                registros_15min=("registros_15min", "sum")
            )
        )
    else:
        resultado = pd.DataFrame()

    print(f"Registros leídos: {registros_leidos}")
    print(f"Registros válidos tras limpieza: {registros_validos}")
    print(f"Registros horarios generados: {len(resultado)}")

    resumen = {
        "archivo": archivo.name,
        "registros_leidos": registros_leidos,
        "registros_validos": registros_validos,
        "registros_horarios": len(resultado)
    }

    return resultado, resumen


# =========================
# PROCESO PRINCIPAL
# =========================

def main():
    print("=" * 100)
    print("INICIO DEL PROCESO DE INGESTA Y LIMPIEZA")
    print("=" * 100)

    archivos_historico = sorted(RUTA_HISTORICO.glob("*.csv"))

    if not archivos_historico:
        raise FileNotFoundError("No se han encontrado CSV en datos/originales/historico_trafico")

    print(f"Archivos históricos encontrados: {len(archivos_historico)}")

    puntos = leer_puntos_medida()

    ids_seleccionados = seleccionar_top_puntos(archivos_historico[0])

    resultados = []
    resumenes = []

    for archivo in archivos_historico:
        df_mes, resumen = procesar_archivo_historico(archivo, ids_seleccionados)

        if not df_mes.empty:
            resultados.append(df_mes)

        resumenes.append(resumen)

    print("\nUniendo todos los meses procesados...")

    trafico = pd.concat(resultados, ignore_index=True)

    # Crear variables temporales
    trafico["año"] = trafico["fecha_hora"].dt.year
    trafico["mes"] = trafico["fecha_hora"].dt.month
    trafico["dia"] = trafico["fecha_hora"].dt.day
    trafico["hora"] = trafico["fecha_hora"].dt.hour
    trafico["dia_semana"] = trafico["fecha_hora"].dt.dayofweek
    trafico["es_fin_semana"] = trafico["dia_semana"].isin([5, 6]).astype(int)

    # Unir coordenadas
    trafico = trafico.merge(
        puntos,
        on="id",
        how="left"
    )

    # Ordenar
    trafico = trafico.sort_values(["id", "fecha_hora"])

    # Exportar dataset limpio
    salida_trafico = RUTA_PROCESADOS / "trafico_limpio.csv"
    trafico.to_csv(salida_trafico, index=False, sep=";")

    # Exportar resumen de limpieza
    resumen_limpieza = pd.DataFrame(resumenes)
    resumen_limpieza.to_csv(RUTA_PROCESADOS / "resumen_limpieza.csv", index=False, sep=";")

    print("\n" + "=" * 100)
    print("PROCESO FINALIZADO")
    print("=" * 100)
    print(f"Archivo generado: {salida_trafico}")
    print(f"Registros finales: {len(trafico)}")
    print(f"Puntos de medida finales: {trafico['id'].nunique()}")
    print(f"Periodo inicial: {trafico['fecha_hora'].min()}")
    print(f"Periodo final: {trafico['fecha_hora'].max()}")
    print("También se ha generado: datos/procesados/resumen_limpieza.csv")
    print("También se ha generado: datos/procesados/puntos_medida_limpios.csv")
    print("También se ha generado: datos/procesados/puntos_seleccionados_modelo.csv")


if __name__ == "__main__":
    main()