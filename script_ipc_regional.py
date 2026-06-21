import pandas as pd
import os

# =============================================================================
# ipc trimestral y regioan INDEC (tomamos como base diciembre 2016 = 100)
#
# los datos los sacamos del archivo indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv
# link: https://infra.datos.gob.ar/catalog/sspm/dataset/145/distribution/145.3/download/indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv
#
# formosa  (aglomerado 15) → usa region NEA → es representada por la columna ipc_ng_nea
# ushuaia  (aglomerado 31) → usa region Patagonia → es representada por la columna ipc_ng_patagonia
#
# La eph es trimestral, entonces se promedian los 3 meses de cada trimestre.
# para el factor de deflactación hacemos: factor = ipc_trimestre / 100
# Para deflactar hacemos: ingreso_real = ingreso_nominal / factor
# Los valores reales quedan expresados en pesos de diciembre 2016. Mas que nada para mantener consistencia a lo largo de los años
# =============================================================================

RUTA_IPC_MENSUAL = "indice-precios-al-consumidor-nivel-general-base-diciembre-2016-mensual.csv"
RUTA_SALIDA = "data/ipc_trimestral_regional.csv"

df = pd.read_csv(RUTA_IPC_MENSUAL, parse_dates=["indice_tiempo"])

df = df[["indice_tiempo", "ipc_ng_nea", "ipc_ng_patagonia"]].copy()
df[["ipc_ng_nea", "ipc_ng_patagonia"]] = df[["ipc_ng_nea", "ipc_ng_patagonia"]].apply(
    pd.to_numeric, errors="coerce"
)

# para extraer año y trimestre desde la fecha mensual
df["ANO4"] = df["indice_tiempo"].dt.year
df["MES"] = df["indice_tiempo"].dt.month
df["TRIMESTRE"] = ((df["MES"] - 1) // 3) + 1

# para promediar los 3 meses de cada trimestre
df_trim = (
    df.groupby(["ANO4", "TRIMESTRE"])[["ipc_ng_nea", "ipc_ng_patagonia"]]
    .mean()
    .reset_index()
    .rename(columns={
        "ipc_ng_nea": "ipc_nea",
        "ipc_ng_patagonia":"ipc_patagonia"
    })
)

# para filtrar solo el periodo EPH
df_trim = df_trim[
    (df_trim["ANO4"] >= 2016) & (df_trim["ANO4"] <= 2025)].copy()

# el primer trimestre disponible es recien el 2016-T4 (la serie arranca en dic-2016), los anteriores no están disponibles
# para los trimestres 2 y 3 de 2016 no hay datos regionales, osea, van a quedar en NaN 
sin_dato = df_trim[df_trim["ipc_nea"].isna()]
if not sin_dato.empty:
    print(f"Trimestres sin IPC regional (fuera de cobertura):\n{sin_dato[['ANO4','TRIMESTRE']]}\n")

# factores de deflactación
df_trim["factor_nea"] = df_trim["ipc_nea"] / 100
df_trim["factor_patagonia"] = df_trim["ipc_patagonia"] / 100

os.makedirs("data", exist_ok=True)
df_trim.to_csv(RUTA_SALIDA, index=False)

print(df_trim.to_string(index=False))
print(f"\nGuardado en: {RUTA_SALIDA}")
