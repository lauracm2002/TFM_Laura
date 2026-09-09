from pathlib import Path
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================

BASE = Path(__file__).resolve().parent.parent
RUTA_PROCESADOS = BASE / "datos" / "procesados" / "definitivos"
RUTA_POWERBI = RUTA_PROCESADOS / "powerbi"
RUTA_POWERBI.mkdir(parents=True, exist_ok=True)

# =========================
# CARGA DE ARCHIVOS
# =========================

print("=" * 100)
print("PREPARANDO ARCHIVOS PARA POWER BI")
print("=" * 100)

def main():
    trafico = pd.read_csv(RUTA_PROCESADOS / "trafico_limpio.csv", sep=";")
    predicciones = pd.read_csv(RUTA_PROCESADOS / "predicciones.csv", sep=";")
    metricas = pd.read_csv(RUTA_PROCESADOS / "metricas_modelo.csv", sep=";")
    importancias = pd.read_csv(RUTA_PROCESADOS / "importancia_variables.csv", sep=";")
    resumen_limpieza = pd.read_csv(RUTA_PROCESADOS / "resumen_limpieza.csv", sep=";")
    puntos = pd.read_csv(RUTA_PROCESADOS / "puntos_medida_limpios.csv", sep=";")
    resumen_modelo = pd.read_csv(RUTA_PROCESADOS / "resumen_modelo.csv", sep=";")

    # =========================
    # PREPARACIÓN DE FECHAS
    # =========================

    trafico["fecha_hora"] = pd.to_datetime(trafico["fecha_hora"]) #Convierte de texto a datatime
    predicciones["fecha_hora"] = pd.to_datetime(predicciones["fecha_hora"])

    trafico["fecha"] = trafico["fecha_hora"].dt.date #Crea una columna fecha sin la hora
    predicciones["fecha"] = predicciones["fecha_hora"].dt.date

    # =========================
    # TABLA RESUMEN GENERAL
    # =========================

    resumen_general = pd.DataFrame([ #Dataframe a partir de una lista de diccionarios
        {"indicador": "Registros dataset limpio", "valor": len(trafico)}, #Cuenta las filas del dataset
        {"indicador": "Puntos de medida analizados", "valor": trafico["id"].nunique()}, #Cuenta los id distintos de sensor
        {"indicador": "Fecha inicial", "valor": str(trafico["fecha_hora"].min())}, #Obtiene primer y última fecha y lo convierte a texto
        {"indicador": "Fecha final", "valor": str(trafico["fecha_hora"].max())},
        {"indicador": "Registros de entrenamiento 2023",
        "valor": resumen_modelo.loc[resumen_modelo["conjunto"] == "entrenamiento_final", #Crea un filtro que localiza la fila del entrenamiento 
        "registros"].iloc[0]}, #Extrae el primer valor del resultado
        {"indicador": "Registros de prueba 2024", #Igual, pero para el conjunto de prueba 2024
        "valor": resumen_modelo.loc[resumen_modelo["conjunto"] == "prueba",
        "registros"].iloc[0]}
    ])

    # =========================
    # TABLA DE CALIDAD DE DATOS
    # =========================

    calidad_datos = pd.DataFrame({ #Crea una tabla que analiza los nulos en cada columna
        "columna": trafico.columns,
        "nulos": trafico.isna().sum().values,
        "porcentaje_nulos": trafico.isna().sum().values / len(trafico) * 100
    })

    # =========================
    # TABLA TEMPORAL AGREGADA
    # =========================

    trafico_temporal = trafico.groupby( 
        ["año", "mes", "dia_semana", "hora"], as_index=False
    ).agg( #Calcula indicadores
        intensidad_media=("intensidad_horaria", "mean"),
        intensidad_total=("intensidad_horaria", "sum"),
        ocupacion_media=("ocupacion_media", "mean"),
        velocidad_media=("velocidad_media", "mean"),
        registros=("id", "count"))

    # =========================
    # TABLA POR PUNTO DE MEDIDA
    # =========================

    trafico_puntos = trafico.groupby( #Tabla que agrupa los datos de cada sensor
        ["id", "nombre", "distrito", "latitud", "longitud"], as_index=False
    ).agg(
        intensidad_media=("intensidad_horaria", "mean"),
        intensidad_total=("intensidad_horaria", "sum"),
        ocupacion_media=("ocupacion_media", "mean"),
        velocidad_media=("velocidad_media", "mean"),
        registros=("fecha_hora", "count"))

    # =========================
    # TABLA DE PREDICCIONES REDUCIDA
    # =========================

    columnas_pred = [ #Columnas que se exportan de la tabla de predicciones
        "id", "fecha_hora", "fecha", "año", "mes", "dia", "hora",
        "dia_semana", "es_fin_semana", "intensidad_horaria",
        "prediccion_persistencia", "prediccion_baseline", #Predicción hora anterior, predicción mismo hora día anterior
        "prediccion_rf", "error_rf", "error_abs_rf", #Predicción generada por Random Forest
        "latitud", "longitud", "nombre", "distrito"
    ]

    predicciones[[c for c in columnas_pred if c in predicciones]]

    # =========================
    # EXPORTACIÓN
    # =========================

    tablas = {
        "trafico_powerbi.csv": trafico,
        "trafico_temporal_powerbi.csv": trafico_temporal,
        "trafico_puntos_powerbi.csv": trafico_puntos,
        "predicciones_powerbi.csv": predicciones[[c for c in columnas_pred if c in predicciones]],
        "metricas_powerbi.csv": metricas,
        "importancia_variables_powerbi.csv": importancias,
        "resumen_limpieza_powerbi.csv": resumen_limpieza,
        "resumen_general_powerbi.csv": resumen_general,
        "calidad_datos_powerbi.csv": calidad_datos,
        "puntos_medida_powerbi.csv": puntos
    }
    for nombre, tabla in tablas.items(): #Bucle de exportación
        tabla.to_csv(RUTA_POWERBI / nombre, index=False, sep=";", #Guarda cada dataframe como csv
                     encoding="utf-8-sig", decimal=",")
    print("Exportaciones generadas en:", RUTA_POWERBI)

if __name__ == "__main__":
    main()
