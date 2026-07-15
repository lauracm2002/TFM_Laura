from pathlib import Path
import pandas as pd

BASE = Path(".")

HISTORICO = BASE / "datos" / "originales" / "historico_trafico"
PUNTOS = BASE / "datos" / "originales" / "puntos_medida"


def leer_csv_prueba(ruta):
    intentos = [
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "latin-1"},
        {"sep": ",", "encoding": "utf-8"},
        {"sep": ",", "encoding": "latin-1"},
    ]

    for intento in intentos:
        try:
            df = pd.read_csv(ruta, nrows=5, **intento)
            return df, intento
        except Exception:
            pass

    raise ValueError(f"No se pudo leer el archivo: {ruta}")


def inspeccionar_carpeta(nombre, carpeta):
    print("\n" + "=" * 100)
    print(f"INSPECCIÓN DE: {nombre}")
    print("=" * 100)

    if not carpeta.exists():
        print(f"No existe la carpeta: {carpeta}")
        return

    archivos = (
        list(carpeta.rglob("*.csv"))
        + list(carpeta.rglob("*.xlsx"))
        + list(carpeta.rglob("*.xls"))
    )

    if not archivos:
        print(f"No se han encontrado CSV ni Excel en: {carpeta}")
        return

    print(f"Archivos encontrados: {len(archivos)}")

    for archivo in archivos[:5]:
        print("\n" + "-" * 100)
        print(f"Archivo: {archivo.name}")
        print(f"Ruta: {archivo}")

        try:
            if archivo.suffix.lower() == ".csv":
                df, config = leer_csv_prueba(archivo)
                print(f"Lectura CSV con: {config}")
            else:
                df = pd.read_excel(archivo, nrows=5)
                print("Lectura Excel correcta")

            print("\nColumnas:")
            print(list(df.columns))

            print("\nPrimeras filas:")
            print(df.head())

        except Exception as e:
            print(f"ERROR leyendo {archivo.name}: {e}")


inspeccionar_carpeta("Histórico de tráfico", HISTORICO)
inspeccionar_carpeta("Puntos de medida", PUNTOS)