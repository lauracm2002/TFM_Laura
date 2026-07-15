from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIGURACIÓN
# =========================

BASE = Path(".")
RUTA_PROCESADOS = BASE / "datos" / "procesados"

ARCHIVO_ENTRADA = RUTA_PROCESADOS / "trafico_limpio.csv"
ARCHIVO_MODELO = RUTA_PROCESADOS / "trafico_modelo.csv"
ARCHIVO_PREDICCIONES = RUTA_PROCESADOS / "predicciones.csv"
ARCHIVO_METRICAS = RUTA_PROCESADOS / "metricas_modelo.csv"

# =========================
# CARGA DE DATOS
# =========================

print("=" * 100)
print("INICIO DEL MODELO PREDICTIVO")
print("=" * 100)

print(f"Leyendo dataset limpio: {ARCHIVO_ENTRADA}")
df = pd.read_csv(ARCHIVO_ENTRADA, sep=";")

df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")

df = df.dropna(subset=["fecha_hora", "id", "intensidad_horaria"])
df = df.sort_values(["id", "fecha_hora"])

print(f"Registros leídos: {len(df)}")
print(f"Puntos de medida: {df['id'].nunique()}")
print(f"Periodo inicial: {df['fecha_hora'].min()}")
print(f"Periodo final: {df['fecha_hora'].max()}")

# =========================
# CREACIÓN DE VARIABLES
# =========================

print("\nCreando variables predictivas...")

df["intensidad_lag_1h"] = df.groupby("id")["intensidad_horaria"].shift(1)
df["intensidad_lag_24h"] = df.groupby("id")["intensidad_horaria"].shift(24)

df["media_movil_3h"] = (
    df.groupby("id")["intensidad_horaria"]
    .shift(1)
    .rolling(window=3)
    .mean()
    .reset_index(level=0, drop=True)
)

df["media_movil_24h"] = (
    df.groupby("id")["intensidad_horaria"]
    .shift(1)
    .rolling(window=24)
    .mean()
    .reset_index(level=0, drop=True)
)

# Eliminar filas sin variables históricas
df_modelo = df.dropna(subset=[
    "intensidad_lag_1h",
    "intensidad_lag_24h",
    "media_movil_3h",
    "media_movil_24h"
]).copy()

print(f"Registros disponibles para modelo: {len(df_modelo)}")

# =========================
# VARIABLES DEL MODELO
# =========================

features = [
    "id",
    "año",
    "mes",
    "dia",
    "hora",
    "dia_semana",
    "es_fin_semana",
    "ocupacion_media",
    "carga_media",
    "velocidad_media",
    "registros_15min",
    "intensidad_lag_1h",
    "intensidad_lag_24h",
    "media_movil_3h",
    "media_movil_24h"
]

# Nos quedamos solo con las columnas que existan
features = [col for col in features if col in df_modelo.columns]

target = "intensidad_horaria"

df_modelo = df_modelo.dropna(subset=features + [target])

# =========================
# DIVISIÓN TEMPORAL
# =========================

# Entrenamiento: año 2023
# Prueba: año 2024
train = df_modelo[df_modelo["año"] == 2023].copy()
test = df_modelo[df_modelo["año"] == 2024].copy()

print(f"Registros entrenamiento 2023: {len(train)}")
print(f"Registros prueba 2024: {len(test)}")

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

# =========================
# MODELO BASE
# =========================

print("\nEvaluando modelo base...")

# Modelo base: predice la intensidad de la misma hora del día anterior
baseline_pred = test["intensidad_lag_24h"]

mae_base = mean_absolute_error(y_test, baseline_pred)
rmse_base = np.sqrt(mean_squared_error(y_test, baseline_pred))
r2_base = r2_score(y_test, baseline_pred)

# =========================
# MODELO RANDOM FOREST
# =========================

print("Entrenando Random Forest...")

modelo_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=14,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

modelo_rf.fit(X_train, y_train)

pred_rf = modelo_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf))
r2_rf = r2_score(y_test, pred_rf)

# =========================
# EXPORTAR MÉTRICAS
# =========================

metricas = pd.DataFrame([
    {
        "modelo": "Baseline lag 24h",
        "MAE": mae_base,
        "RMSE": rmse_base,
        "R2": r2_base
    },
    {
        "modelo": "Random Forest Regressor",
        "MAE": mae_rf,
        "RMSE": rmse_rf,
        "R2": r2_rf
    }
])

metricas.to_csv(ARCHIVO_METRICAS, index=False, sep=";")

print("\nMétricas del modelo:")
print(metricas)

# =========================
# EXPORTAR PREDICCIONES
# =========================

predicciones = test.copy()

predicciones["prediccion_baseline"] = baseline_pred.values
predicciones["prediccion_rf"] = pred_rf
predicciones["error_rf"] = predicciones["intensidad_horaria"] - predicciones["prediccion_rf"]
predicciones["error_abs_rf"] = predicciones["error_rf"].abs()

# Para Power BI y memoria, no necesitamos demasiadas columnas
columnas_predicciones = [
    "id",
    "fecha_hora",
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

columnas_predicciones = [c for c in columnas_predicciones if c in predicciones.columns]

predicciones[columnas_predicciones].to_csv(
    ARCHIVO_PREDICCIONES,
    index=False,
    sep=";"
)

# Exportar dataset usado para el modelo
df_modelo.to_csv(ARCHIVO_MODELO, index=False, sep=";")

# =========================
# IMPORTANCIA DE VARIABLES
# =========================

importancias = pd.DataFrame({
    "variable": features,
    "importancia": modelo_rf.feature_importances_
}).sort_values("importancia", ascending=False)

importancias.to_csv(
    RUTA_PROCESADOS / "importancia_variables.csv",
    index=False,
    sep=";"
)

print("\nVariables más importantes:")
print(importancias.head(10))

print("\n" + "=" * 100)
print("MODELO FINALIZADO")
print("=" * 100)
print(f"Archivo generado: {ARCHIVO_MODELO}")
print(f"Archivo generado: {ARCHIVO_PREDICCIONES}")
print(f"Archivo generado: {ARCHIVO_METRICAS}")
print(f"Archivo generado: {RUTA_PROCESADOS / 'importancia_variables.csv'}")