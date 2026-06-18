import pandas as pd
import glob
import os

# configuracion de carpetas
carpeta_datos = "./datos_indec"  # Nombre de carpeta unificado
ruta_salida = "data/eph_formosa_ushuaia_2016_2025.csv"


COLUMNAS_NECESARIAS = {
    'ANO4', 'TRIMESTRE', 'AGLOMERADO', 'PONDERA',
    'ESTADO', 'CAT_OCUP', 'PP04B_COD', 'PP04D_COD',
    'P21', 'P47T', 'CH04', 'CH06', 'NIVEL_ED'
}
AGLOMERADOS = ['15', '31']  # 15 representa a Formosa y 31 a Ushuaia, Tierra del fuego

archivos_eph = glob.glob(os.path.join(carpeta_datos, "individual_*.csv"))
lista_recortes = []
archivos_con_error = []

print(f"Se encontraron {len(archivos_eph)} archivos para procesar en '{carpeta_datos}'...")

for i, archivo in enumerate(archivos_eph, 1):
    nombre = os.path.basename(archivo)
    print(f"[{i}/{len(archivos_eph)}] Procesando: {nombre}")
    exito = False

    # combinaciones de separadores y encodings por la variabilidad del INDEC en el paso del tiempo
    for sep in [';', ',', '\t']:
        for encoding in ['utf-8', 'latin-1']:
            try:
                # Lectura de cabecera para validar columnas sin cargar todo
                df_hdr = pd.read_csv(archivo, sep=sep, encoding=encoding, nrows=0)
                df_hdr.columns = df_hdr.columns.str.upper().str.strip()
                if not COLUMNAS_NECESARIAS.issubset(df_hdr.columns):
                    continue
                
                df_temp = pd.read_csv(
                    archivo, sep=sep, encoding=encoding, dtype=str,
                    usecols=lambda c: c.upper().strip() in COLUMNAS_NECESARIAS
                )
                df_temp.columns = df_temp.columns.str.upper().str.strip()
                df_recortado = df_temp[df_temp['AGLOMERADO'].isin(AGLOMERADOS)].copy()

                lista_recortes.append(df_recortado)
                print(f" -> {len(df_recortado)} filas (sep='{sep}', enc='{encoding}')")
                exito = True
                break
            except Exception:
                continue
        if exito:
            break

    if not exito:
        print(f"error: no se pudo procesar {nombre}")
        archivos_con_error.append(nombre)

if archivos_con_error:
    print(f"\narchivos con error ({len(archivos_con_error)}):")
    for f in archivos_con_error:
        print(f"- {f}")

if lista_recortes:
    df_historico_raw = pd.concat(lista_recortes, ignore_index=True)
    df_historico_raw = df_historico_raw.sort_values(['ANO4', 'TRIMESTRE']).reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df_historico_raw.to_csv(ruta_salida, index=False)

    print(f"\nbase guardada en: {ruta_salida}")
    print(f"filas totales: {len(df_historico_raw)}")
    print(f"columnas: {list(df_historico_raw.columns)}")
else:
    print("error: no se pudo crear ningun archivo")