from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# CONFIGURACIÓN GENERAL
# =========================

BASE = Path(__file__).resolve().parent.parent
RUTA_HISTORICO = BASE / "datos" / "originales" / "historico_trafico"
RUTA_PUNTOS = BASE / "datos" / "originales" / "puntos_medida"
RUTA_PROCESADOS = BASE / "datos" / "procesados" / "definitivos"
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
MIN_REGISTROS_HORA = 4

# =========================
# FUNCIONES AUXILIARES
# =========================

#Función que lee el Excel de coordenadas y prepara la información geográfica de los sensores
def leer_puntos_medida():
    """
    Lee el archivo Excel de puntos de medida y devuelve una tabla limpia
    con id, nombre, distrito, longitud y latitud.
    """
    archivos = sorted(RUTA_PUNTOS.glob("*.xlsx")) + sorted(RUTA_PUNTOS.glob("*.xls")) #se ordenan los archivos

    if not archivos:
        raise FileNotFoundError(f"No se ha encontrado ningún archivo Excel en {RUTA_PUNTOS}")

    archivo = archivos[0]
    print(f"Leyendo puntos de medida: {archivo.name}")
    puntos = pd.read_excel(archivo)

    puntos.columns = (
        puntos.columns
        .str.strip() #espacios en blanco
        .str.lower() #minúsculas
        .str.replace(" ", "_") #espacios por guiones
    )

    columnas_necesarias = ["id", "tipo_elem", "distrito", "nombre", "longitud", "latitud"]
    columnas_existentes = [c for c in columnas_necesarias if c in puntos.columns]
    puntos = puntos[columnas_existentes].copy() #copia las columnas existentes

    puntos["id"] = pd.to_numeric(puntos["id"], errors="coerce") #convierte id a número
    puntos = puntos.dropna(subset=["id"]) #elimina filas sin id
    puntos = puntos.drop_duplicates(subset=["id"]) #elimina duplicados del sensor
    puntos["id"] = puntos["id"].astype(int) #convierte id a entero

    if "longitud" in puntos.columns: #si existe longitud en las columnas, convierte a número
        puntos["longitud"] = pd.to_numeric(puntos["longitud"], errors="coerce")

    if "latitud" in puntos.columns: #si existe ltitud en las columnas, convierte a número
        puntos["latitud"] = pd.to_numeric(puntos["latitud"], errors="coerce")

    puntos.to_csv(RUTA_PROCESADOS / "puntos_medida_limpios.csv", index=False, sep=";") #guarda la tabla limpia

    print(f"Puntos de medida limpios: {len(puntos)}")
    return puntos

#Función que escoge los 50 sensores con más registros marcados como válidos
def seleccionar_top_puntos(archivo_referencia):
    """
    Selecciona los puntos con más registros válidos en el primer archivo.
    """
    print(f"Seleccionando los {MAX_PUNTOS} puntos con más registros usando: {archivo_referencia.name}")

    conteos = {} #guarda los registros válidos de cada sensor

    for chunk in pd.read_csv( #lee el CSV por bloques de 500.000 filas
        archivo_referencia,
        sep=";",
        encoding="utf-8",
        usecols=["id", "error"], #limita la lectura a id y error
        chunksize=CHUNKSIZE
    ):
        chunk = chunk[chunk["error"] #filtra registros cuyo indicador de error es N
        .astype(str) #convierte valores a texto
        .str.strip() #elimina espacios
        .str.upper() == "N"] #mayúsculas
        vc = pd.to_numeric(chunk["id"], errors="coerce").dropna().astype(int).value_counts()

        for id_punto, n in vc.items(): #recorre los conteos del bloque
            conteos[id_punto] = conteos.get(id_punto, 0) + int(n) #recupera el acumulado anterior y suma los registros del bloque actual

    serie = pd.Series(conteos, dtype="int64").sort_values(ascending=False) #ordena de mayor a menor
    top_ids = serie.head(MAX_PUNTOS).index.astype(int).tolist() #selecciona los 50 primeros

    print(f"Puntos seleccionados: {len(top_ids)}")
    print(f"Primeros puntos seleccionados: {top_ids[:10]}")

    pd.DataFrame({ #crea y guarda un csv con los sensores seleccionados y sus conteos
        "id": top_ids,
        "registros_archivo_referencia": serie.head(MAX_PUNTOS).values
    }).to_csv(RUTA_PROCESADOS / "puntos_seleccionados_modelo.csv", index=False, sep=";")

    return top_ids

#Función que recibe un csv y devuelve un resumen de calidad
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

    grupos_archivo = [] #lista que guarda los bloques filtrados
    registros_leidos = 0 #cuenta las filas originales
    registros_validos = 0 #cuenta las filas que superen los filtros

    for chunk in pd.read_csv( #lee el archivo por bloques
        archivo,
        sep=";",
        encoding="utf-8",
        usecols=COLUMNAS_HISTORICO,
        chunksize=CHUNKSIZE
    ):
        registros_leidos += len(chunk) #devuelve el número de filas por bloque y lo suma a lo acumulado

        chunk = chunk[chunk["id"].isin(ids_seleccionados)].copy() #Comprueba si el id está entre los sensores seleccionados

        chunk = chunk[chunk["error"].astype(str).str.strip().str.upper() == "N"].copy() #Filtra registros sin error

        chunk["fecha"] = pd.to_datetime(chunk["fecha"], errors="coerce") #Convierte le fecha a tipo datatime

        for col in ["intensidad", "ocupacion", "carga", "vmed", "periodo_integracion"]: #Convierte en numéricas las variables
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        chunk = chunk.dropna(subset=["id", "fecha", "intensidad"]) #Elimina filas sin id, fecha e intensidad
        chunk = chunk[chunk["intensidad"] >= 0].copy() #Descarta las intensidades negativas

        registros_validos += len(chunk) #Suma al contador los registros que han pasado el filtro

        grupos_archivo.append(chunk[["id", "fecha", "intensidad", "ocupacion",
        "carga", "vmed", "periodo_integracion"]].copy()) #Añade a la lista de bloques los necesarios

    if not grupos_archivo:
        return pd.DataFrame(), {"archivo": archivo.name, "registros_leidos": registros_leidos,
        "registros_validos": 0, "registros_horarios": 0} #Devuelve un dataframe vacío si no se ha añadido ningún bloque

    datos = pd.concat(grupos_archivo, ignore_index=True) #Une los bloques filtrados

    duplicados = int(datos.duplicated(["id", "fecha"]).sum()) #Elimina duplicados de sensor y fecha
    if duplicados: #Si existen duplicados el programa se detiene
        raise ValueError(f"{archivo.name}: {duplicados} fechas duplicadas por sensor; "
        "revisar antes de agregar.")

    datos["fecha_hora"] = datos["fecha"].dt.floor("h") #Redondea cada hora a su comienzo

    grupo = datos.groupby(["id", "fecha_hora"], as_index=False).agg( #Agrupa las filas que pertenecen al mismo sensor y misma hora
        intensidad_horaria=("intensidad", "mean"), #Se definen agregaciones
        ocupacion_media=("ocupacion", "mean"),
        carga_media=("carga", "mean"),
        velocidad_media=("vmed", "mean"),
        periodo_integracion_medio=("periodo_integracion", "mean"),
        registros_15min=("intensidad", "count")
    )

    cuartos = datos.groupby(["id", "fecha_hora"])["fecha"].nunique() #Cuenta los timestamps distintos
    if (cuartos > 4).any(): #Sale error si alguna hora tiene más de 4 timestamps distintos
        raise ValueError(f"{archivo.name}: más de cuatro timestamps por hora; revisar frecuencia.")

    incompletas = int((grupo["registros_15min"] != MIN_REGISTROS_HORA).sum()) #Cuenta las horas que no tienen cuatro registros
    grupo = grupo[grupo["registros_15min"] == MIN_REGISTROS_HORA].copy() #Conserva las horas completas

    minutos = datos["fecha"].dt.minute #Comprueba que el minuto está entre o, 15, 30 y 45
    if not minutos.isin([0, 15, 30, 45]).all():
        raise ValueError(f"{archivo.name}: timestamps fuera de los cuartos esperados.")

    print(f"Registros leídos: {registros_leidos}")
    print(f"Registros válidos tras limpieza: {registros_validos}")
    print(f"Registros horarios generados: {len(grupo)}")

    resumen = { #Diccionario con las estadísticas del mes
        "archivo": archivo.name,
        "registros_leidos": registros_leidos,
        "registros_validos": registros_validos,
        "registros_horarios": len(grupo),
        "horas_incompletas_excluidas": incompletas
    }
    return grupo, resumen

# =========================
# PROCESO PRINCIPAL
# =========================

def main():
    print("=" * 100)
    print("INICIO DEL PROCESO DE INGESTA Y LIMPIEZA")
    print("=" * 100)

    archivos_historico = sorted(RUTA_HISTORICO.glob("*.csv")) #Ordena todos los archivos

    if not archivos_historico: #Si no los encuentra, sale error
        raise FileNotFoundError(f"No se han encontrado CSV en {RUTA_HISTORICO}")
    print(f"Archivos históricos encontrados: {len(archivos_historico)}")

    puntos = leer_puntos_medida() #Utiliza el primer csv ordenado
    ids_seleccionados = seleccionar_top_puntos(archivos_historico[0])

    resultados = []
    resumenes = []

    for archivo in archivos_historico: #Recorre los archivos procesados y los muestra
        print("Procesando:", archivo.name, flush=True)
        mes, resumen = procesar_archivo_historico(archivo, ids_seleccionados)

        if not mes.empty: #Si no está vacío el dataframe, lo añade a la lista de resultados
            resultados.append(mes)
        resumenes.append(resumen)
        print(resumen, flush=True)

    if not resultados: #Si no hay datos, da error
        raise ValueError("No se han generado registros válidos.")

    print("\nUniendo todos los meses procesados...")

    trafico = pd.concat(resultados, ignore_index=True) #Une los dataframes en una tabla

    if trafico.duplicated(["id", "fecha_hora"]).any(): #Comprueba que no hayan mismas horas en un mismo sensor
        raise ValueError("Existen horas duplicadas entre archivos mensuales.")

    trafico["año"] = trafico["fecha_hora"].dt.year #Se crean variables temporales
    trafico["mes"] = trafico["fecha_hora"].dt.month
    trafico["dia"] = trafico["fecha_hora"].dt.day
    trafico["hora"] = trafico["fecha_hora"].dt.hour
    trafico["dia_semana"] = trafico["fecha_hora"].dt.dayofweek
    trafico["es_fin_semana"] = trafico["dia_semana"].isin([5, 6]).astype(int)

    trafico = trafico.merge( #Se unen las coordenadas mediante la unión
        puntos,
        on="id",
        how="left",
        validate="many_to_one"
    )

    trafico = trafico.sort_values(["id", "fecha_hora"]) #Ordena el dataset por sensor y por fecha

    trafico.to_csv(RUTA_PROCESADOS / "trafico_limpio.csv", index=False, sep=";") # Exporta el dataset limpio

    pd.DataFrame(resumenes).to_csv(RUTA_PROCESADOS / "resumen_limpieza.csv", index=False, sep=";") #Convierte resúmenes en una tabla

    cobertura = trafico.groupby(["id", "año", "mes"]).size().rename("horas_observadas").reset_index() #Agrupa los datos
    cobertura.to_csv(RUTA_PROCESADOS / "cobertura_mensual.csv", index=False, sep=";")
    print("Dataset generado:", len(trafico), "filas")
    print("Salida:", RUTA_PROCESADOS)

    print("\n" + "=" * 100)
    print("PROCESO FINALIZADO")
    print("=" * 100)

if __name__ == "__main__":
    main()