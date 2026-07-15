from pathlib import Path
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

BASE = Path(".")
RUTA_PROCESADOS = BASE / "datos" / "procesados"
RUTA_POWERBI = RUTA_PROCESADOS / "powerbi"

RUTA_POWERBI.mkdir(parents=True, exist_ok=True)

# =========================
# CARGA DE ARCHIVOS
# =========================

print("=" * 100)
print("PREPARANDO ARCHIVOS PARA POWER BI")
print("=" * 100)

trafico = pd.read_csv(RUTA_PROCESADOS / "trafico_limpio.csv", sep=";")
predicciones = pd.read_csv(RUTA_PROCESADOS / "predicciones.csv", sep=";")
metricas = pd.read_csv(RUTA_PROCESADOS / "metricas_modelo.csv", sep=";")
importancias = pd.read_csv(RUTA_PROCESADOS / "importancia_variables.csv", sep=";")
resumen_limpieza = pd.read_csv(RUTA_PROCESADOS / "resumen_limpieza.csv", sep=";")
puntos = pd.read_csv(RUTA_PROCESADOS / "puntos_medida_limpios.csv", sep=";")

# =========================
# PREPARACIÓN DE FECHAS
# =========================

trafico["fecha_hora"] = pd.to_datetime(trafico["fecha_hora"], errors="coerce")
predicciones["fecha_hora"] = pd.to_datetime(predicciones["fecha_hora"], errors="coerce")

trafico["fecha"] = trafico["fecha_hora"].dt.date
predicciones["fecha"] = predicciones["fecha_hora"].dt.date

# =========================
# TABLA RESUMEN GENERAL
# =========================

resumen_general = pd.DataFrame([
    {
        "indicador": "Registros dataset limpio",
        "valor": len(trafico)
    },
    {
        "indicador": "Puntos de medida analizados",
        "valor": trafico["id"].nunique()
    },
    {
        "indicador": "Fecha inicial",
        "valor": str(trafico["fecha_hora"].min())
    },
    {
        "indicador": "Fecha final",
        "valor": str(trafico["fecha_hora"].max())
    },
    {
        "indicador": "Registros de entrenamiento 2023",
        "valor": len(trafico[trafico["año"] == 2023])
    },
    {
        "indicador": "Registros de prueba 2024",
        "valor": len(trafico[trafico["año"] == 2024])
    }
])

# =========================
# TABLA DE CALIDAD DE DATOS
# =========================

calidad_datos = pd.DataFrame({
    "columna": trafico.columns,
    "nulos": trafico.isna().sum().values,
    "porcentaje_nulos": (trafico.isna().sum().values / len(trafico)) * 100
})

# =========================
# TABLA TEMPORAL AGREGADA
# =========================

trafico_temporal = (
    trafico
    .groupby(["año", "mes", "dia_semana", "hora"], as_index=False)
    .agg(
        intensidad_media=("intensidad_horaria", "mean"),
        intensidad_total=("intensidad_horaria", "sum"),
        ocupacion_media=("ocupacion_media", "mean"),
        velocidad_media=("velocidad_media", "mean"),
        registros=("id", "count")
    )
)

# =========================
# TABLA POR PUNTO DE MEDIDA
# =========================

trafico_puntos = (
    trafico
    .groupby(["id", "nombre", "distrito", "latitud", "longitud"], as_index=False)
    .agg(
        intensidad_media=("intensidad_horaria", "mean"),
        intensidad_total=("intensidad_horaria", "sum"),
        ocupacion_media=("ocupacion_media", "mean"),
        velocidad_media=("velocidad_media", "mean"),
        registros=("fecha_hora", "count")
    )
)

# =========================
# TABLA DE PREDICCIONES REDUCIDA
# =========================

# Para Power BI no hace falta cargar absolutamente todas las columnas.
columnas_pred = [
    "id",
    "fecha_hora",
    "fecha",
    "año",
    "mes",
    "dia",
    "hora",
    "dia_semana",
    "es_fin_semana",
    "intensidad_horaria",
    "prediccion_baseline",
    "prediccion_rf",
    "error_rf",
    "error_abs_rf",
    "latitud",
    "longitud",
    "nombre",
    "distrito"
]

columnas_pred = [c for c in columnas_pred if c in predicciones.columns]
predicciones_powerbi = predicciones[columnas_pred].copy()

# =========================
# EXPORTACIÓN
# =========================

trafico.to_csv(RUTA_POWERBI / "trafico_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
trafico_temporal.to_csv(RUTA_POWERBI / "trafico_temporal_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
trafico_puntos.to_csv(RUTA_POWERBI / "trafico_puntos_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
predicciones_powerbi.to_csv(RUTA_POWERBI / "predicciones_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
metricas.to_csv(RUTA_POWERBI / "metricas_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
importancias.to_csv(RUTA_POWERBI / "importancia_variables_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
resumen_limpieza.to_csv(RUTA_POWERBI / "resumen_limpieza_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
resumen_general.to_csv(RUTA_POWERBI / "resumen_general_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
calidad_datos.to_csv(RUTA_POWERBI / "calidad_datos_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")
puntos.to_csv(RUTA_POWERBI / "puntos_medida_powerbi.csv", index=False, sep=";", encoding="utf-8-sig", decimal=",")

print("Archivos generados en datos/procesados/powerbi:")
for archivo in RUTA_POWERBI.glob("*.csv"):
    print(f"- {archivo.name}")

print("\nProceso finalizado correctamente.")