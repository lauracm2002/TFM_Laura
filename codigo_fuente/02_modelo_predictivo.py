from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# CONFIGURACIÓN
# =========================

BASE = Path(__file__).resolve().parent.parent
RUTA_PROCESADOS = BASE / "datos" / "procesados" / "definitivos"

ARCHIVO_ENTRADA = RUTA_PROCESADOS / "trafico_limpio.csv"
ARCHIVO_MODELO = RUTA_PROCESADOS / "trafico_modelo.csv"
ARCHIVO_PREDICCIONES = RUTA_PROCESADOS / "predicciones.csv"
ARCHIVO_METRICAS = RUTA_PROCESADOS / "metricas_modelo.csv"

print("=" * 100)
print("INICIO DEL MODELO PREDICTIVO")
print("=" * 100)

#Entrenamiento hasta septiembre de 2023
FECHA_VALIDACION = pd.Timestamp("2023-10-01") #Validación de octubre a diciembre de 2023
FECHA_PRUEBA = pd.Timestamp("2024-01-01") #Prueba desde enero de 2024

VARIABLES_OPERATIVAS = ["ocupacion_media", "carga_media", "velocidad_media", "registros_15min"] #Variable para predecir el tráfico

# =========================
# PREPARACIÓN DE LOS DATOS
# =========================

#Función para los datos para que cada variable predictiva respete el tiempo y no se mezclen sensores
def preparar_datos(df):
    df = df.copy()

    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce")

    df = df.dropna(subset=["id", "fecha_hora", "intensidad_horaria"]) #Elimina las filas sin id, fecha e intensidad

    df["id"] = pd.to_numeric(df["id"], errors="coerce") #Convierte a numérico
    df = df.dropna(subset=["id"]) #Elimina los que no son válidos
    df["id"] = df["id"].astype(int) #Los transforma en enteros

    if df.duplicated(["id", "fecha_hora"]).any(): #Comprueba duplicados del mismo sensor a la misma hora
        raise ValueError("Existen duplicados id-hora. Revisa la ingesta.")
    
    df = df.sort_values(["id", "fecha_hora"]) #Ordena por sensor y por fecha

    grupos = []
    for id_punto, grupo in df.groupby("id", sort=False): #Recorre los datos sensor por sensor
        grupo = grupo.set_index("fecha_hora")
        calendario = pd.date_range(grupo.index.min(), grupo.index.max(), freq="h") #Genera todas las horas del sensor
        grupo = grupo.reindex(calendario) #Alinea los datos con el calendario completo
        grupo.index.name = "fecha_hora" #Da nombre al índice
        grupo["id"] = id_punto #Asigna el id del sensor
        grupos.append(grupo.reset_index()) #Añade la fecha convertida en columna a la lista

    df = pd.concat(grupos, ignore_index=True) #Concatena los calendarios de sensores en una tabla
    df = df.sort_values(["id", "fecha_hora"]).reset_index(drop=True)

    y = df.groupby("id")["intensidad_horaria"] #Crea un objeto agrupado por sensor que tiene la columna de intensidad

    df["intensidad_lag_1h"] = y.shift(1) #Desplaza los valores una fila abajo de cada sensor
    df["intensidad_lag_24h"] = y.shift(24)

    for ventana in [3, 24]: #Recorre dos tamaños de ventana: 3 y 24 horas
        df[f"media_movil_{ventana}h"] = (
            df.groupby("id")["intensidad_horaria"] #Separa las series por sensor
            .transform(lambda s: s #Define una función que recibe la serie de intensidad de un sensor
            .shift(1) #Desplaza la serie una hora hacia el pasado
            .rolling(ventana, min_periods=ventana) #Crea ventanas cuyos datos estén disponibles
            .mean()) #Calcula la media de cada ventana
        )

    for col in VARIABLES_OPERATIVAS:
        if col in df: #Comprueba si la columna existe en el dataframe
            df[f"{col}_lag_1h"] = df.groupby("id")[col].shift(1) #Crea columnas con una versión con sufijo de una hora

    df["año"] = df["fecha_hora"].dt.year
    df["mes"] = df["fecha_hora"].dt.month
    df["dia"] = df["fecha_hora"].dt.day
    df["hora"] = df["fecha_hora"].dt.hour
    df["dia_semana"] = df["fecha_hora"].dt.dayofweek
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6]).astype(int) #Convierte la condición de sábado o domingo en 1 o 0

    features = ["id", "mes", "dia", "hora", "dia_semana", "es_fin_semana", #Lista columnas entrada Random Forest
                "intensidad_lag_1h", "intensidad_lag_24h",
                "media_movil_3h", "media_movil_24h"]
    
    features += [f"{c}_lag_1h" for c in VARIABLES_OPERATIVAS if c in df] #Versiones retardadas de variables operativas
    df_modelo = df.dropna(subset=features + ["intensidad_horaria"]).copy() #Elimina filas
    return df_modelo, features, df

# =========================
# CÁLCULO DE MÉTRICAS
# =========================

#Función para el cáculo de métricas 
def calcular_metricas(y_real, y_pred, nombre, conjunto): #Recibe valores
    return {"modelo": nombre, "conjunto": conjunto, #Devuelve un diccionario con las métricas y el número de observaciones
            "MAE": mean_absolute_error(y_real, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_real, y_pred)),
            "R2": r2_score(y_real, y_pred), "n": len(y_real)}

def main():
    print("=" * 80)
    print("INICIO DEL MODELO PREDICTIVO")
    print("=" * 80)

    df = pd.read_csv(ARCHIVO_ENTRADA, sep=";") #Lee el dataset corregido
    df_modelo, features, calendario = preparar_datos(df)

    print("Registros originales:", len(df))
    print("Filas calendario:", len(calendario))
    print("Registros disponibles para modelado:", len(df_modelo))
    print("Variables:", features)

    train = df_modelo[df_modelo["fecha_hora"] < FECHA_VALIDACION].copy() #Selecciona filas antes al 1 octubre

    val = df_modelo[(df_modelo["fecha_hora"] >= FECHA_VALIDACION) & #Filas desde 1 octubre a 31 diciembre
        (df_modelo["fecha_hora"] < FECHA_PRUEBA)].copy()
    
    test = df_modelo[df_modelo["fecha_hora"] >= FECHA_PRUEBA].copy() #Filas desde el 1 de enero 2024

    if train.empty or val.empty or test.empty: #Comprueba que ninguna partición esté vacía
        raise ValueError("Una partición temporal está vacía.")
    
    print("Entrenamiento:", len(train), "Validación:", len(val), "Prueba:", len(test))
    target = "intensidad_horaria" #Define la columna que quiero predecir

    # ===============================================
    # CREACIÓN Y ENTRENAMIENTO RANDOM FOREST
    # ===============================================

    modelo_rf = RandomForestRegressor( #Crea el modelo con los parámetros
        n_estimators=100, max_depth=14, min_samples_leaf=5,
        random_state=42, n_jobs=-1)
    modelo_rf.fit(train[features], train[target]) #Entrena el modelo

    pred_val = modelo_rf.predict(val[features]) #Aplica el modelo al conjunto de validación
    validacion = pd.DataFrame([ #Crea una tabla con las métricas de tres modelos
        calcular_metricas(val[target], val["intensidad_lag_1h"], "Persistencia", "validacion"),
        calcular_metricas(val[target], val["intensidad_lag_24h"], "Baseline lag 24h", "validacion"),
        calcular_metricas(val[target], pred_val, "Random Forest Regressor", "validacion")
    ])
    validacion.to_csv(RUTA_PROCESADOS / "metricas_validacion.csv", index=False, sep=";")

    train_final = pd.concat([train, val], ignore_index=True)
    modelo_rf.fit(train_final[features], train_final[target]) #Vuelve a entrenar el modelo
    pred_rf = modelo_rf.predict(test[features]) #Genera las predicciones sobre 2024

    pred_rf = np.maximum(pred_rf, 0) #Compara cada predicción con 0 y conserva el mayor

    metricas = pd.DataFrame([ #Calcula las métricas definitivas de los tres modelos
        calcular_metricas(test[target], test["intensidad_lag_1h"], "Persistencia", "prueba"),
        calcular_metricas(test[target], test["intensidad_lag_24h"], "Baseline lag 24h", "prueba"),
        calcular_metricas(test[target], pred_rf, "Random Forest Regressor", "prueba")
    ])
    metricas.to_csv(ARCHIVO_METRICAS, index=False, sep=";")
    print(metricas.to_string(index=False))

    predicciones = test.copy()
    predicciones["prediccion_persistencia"] = test["intensidad_lag_1h"] #Añade tres columnas con predicciones
    predicciones["prediccion_baseline"] = test["intensidad_lag_24h"]
    predicciones["prediccion_rf"] = pred_rf

    predicciones["error_rf"] = predicciones[target] - predicciones["prediccion_rf"] #Calcula el error con signo
    predicciones["error_abs_rf"] = predicciones["error_rf"].abs() #Calcula el error absoluto

    columnas = ["id", "fecha_hora", "año", "mes", "dia", "hora", "dia_semana", #Valores para PowerBi
                "es_fin_semana", target, "prediccion_persistencia",
                "prediccion_baseline", "prediccion_rf", "error_rf", "error_abs_rf",
                "latitud", "longitud", "nombre", "distrito"]
    
    predicciones[[c for c in columnas if c in predicciones]].to_csv(
        ARCHIVO_PREDICCIONES, index=False, sep=";")
    
    df_modelo.to_csv(ARCHIVO_MODELO, index=False, sep=";")

    pd.DataFrame({"variable": features, "importancia": modelo_rf.feature_importances_}
                 ).sort_values("importancia", ascending=False).to_csv(
                     RUTA_PROCESADOS / "importancia_variables.csv", index=False, sep=";")
    
    pd.DataFrame([ #Crea un csv con el número de registros de cada partición
        {"conjunto": "entrenamiento", "registros": len(train)},
        {"conjunto": "validacion", "registros": len(val)},
        {"conjunto": "entrenamiento_final", "registros": len(train_final)},
        {"conjunto": "prueba", "registros": len(test)}
    ]).to_csv(RUTA_PROCESADOS / "resumen_modelo.csv", index=False, sep=";")
    print("Resultados guardados en:", RUTA_PROCESADOS)

if __name__ == "__main__":
    main()
