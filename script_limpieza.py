import os
import numpy as np
import pandas as pd

cargar_path = "data/eph_formosa_ushuaia_2016_2025.csv"
guardar_path = "data/eph_formosa_ushuaia_limpio.csv"
ipc_path = "data/ipc_trimestral_regional.csv"

#carga y conversion de tipos
df = pd.read_csv(cargar_path)
columnas_numericas = [
    "ANO4", "TRIMESTRE", "AGLOMERADO", "PONDERA", "ESTADO", "PONDII", "PONDIIO",
    "CAT_OCUP", "PP04B_COD", "PP04D_COD", "P21", "P47T",
    "CH04", "CH06", "NIVEL_ED"
]

df[columnas_numericas] = df[columnas_numericas].apply(pd.to_numeric, errors="coerce")

print(f"Filas cargadas: {len(df)} | período: {int(df['ANO4'].min())} - {int(df['ANO4'].max())}\n")

#mapeo limpio de no respuesta a NaN con replace
mapeo_no_respuesta = {
    'ESTADO': [9],
    'CAT_OCUP': [9],
    'NIVEL_ED': [9],
    'CH04': [9],
    'PP04B_COD': [99, 999, 9999],
    'PP04D_COD': [99, 999, 9999],
    'P21': [-9],
    'P47T': [-9]
}

print("=" * 60 + "\nReporte de no respuesta\n" + "=" * 60)
for columna, codigos in mapeo_no_respuesta.items():
    n = df[columna].isin(codigos).sum()
    pct = 100 * n / len(df)
    print(f'{columna:12s}: {n: 6d} casos ({pct:.1f}%)')
    df[f"{columna}_LIMPIO"] = df[columna].replace(codigos, np.nan)

# limpieza de edad erronea
df["CH06_LIMPIO"] = df["CH06"].where(df["CH06"] >= 0)

# deflactacion o normalización de valores monetario
# Formosa (aglomerado 15) -> usa factor_nea
# Ushuaia (aglomerado 31) -> usa factor_patagonia

# ingreso_real = ingreso_nominal / factor_regional
# El resultado está expresado en pesos de diciembre 2016.

df_ipc = pd.read_csv(ipc_path)
df = pd.merge(df, df_ipc[["ANO4", "TRIMESTRE", "factor_nea", "factor_patagonia"]],
              on=["ANO4", "TRIMESTRE"], how="left")

df["factor_regional"] = np.where(
    df["AGLOMERADO"] == 15,
    df["factor_nea"],
    df["factor_patagonia"]
)

filas_sin_factor = df["factor_regional"].isna().sum()
if filas_sin_factor > 0:
    print(f"\nadvertencia: {filas_sin_factor} filas sin factor IPC (trimestres fuera de rango).")

df["P21_REAL"] = df["P21_LIMPIO"] / df["factor_regional"]
df["P47T_REAL"] = df["P47T_LIMPIO"] / df["factor_regional"]

#diagnostico de outliners

def diagnosticar(columna, nombre):
    columna_limpia = columna.dropna() #metodo de pandas que elimina datos faltantes en un conjunto de datos
    q1, q3 = columna_limpia.quantile([0.25, 0.75]) #el cuartil 1 (q1) representa el 25% de los datos, el cuartil 3 (q3) el 75%
    iqr = q3 - q1 #iqr representa el 50% de un conjunto de datos

    outliers = columna_limpia[(columna_limpia < q1 - 1.5 * iqr) | (columna_limpia > q3 + 1.5 * iqr)]

    print(f"{nombre:15s}: {len(outliers):6d} outliers ({100*len(outliers)/len(columna_limpia):.1f}%) | limites: [{q1 - 1.5 * iqr:.1f}, {q3 + 1.5 * iqr:.1f}]")

print("\n" + ("=" * 60) + "\nEstadística descriptiva — variables continuas\n" + "=" * 60)
#qseleccionamos tres columnas del dataframe para trabajar con ellas
#describe es un metodo de pandas que produce varias estadísticas de resumen de una sola vez. transpose intercambia filas por columnas, es estetico
print(df[["CH06_LIMPIO", "P21_REAL", "P47T_REAL"]].describe().transpose())

print("\n" + ("=" * 60) + "\nComparativa de outliers: nominal vs real\n" + "=" * 60)
for columna in ["P21_LIMPIO", "P21_REAL", "P47T_LIMPIO", "P47T_REAL"]:
    diagnosticar(df[columna], columna)

print("\n" + ("=" * 60) + "\ndistribucion — variables categoricas\n" + "=" * 60)
for columna in ["ESTADO_LIMPIO", "CH04_LIMPIO", "NIVEL_ED_LIMPIO", "CAT_OCUP_LIMPIO"]:
    print(f"\n{columna}:\n", df[columna].value_counts(dropna=False, normalize=True).mul(100).round(1))

# Borrado lógico: trimestres sin IPC regional (2016-T2 y T3)
df = df[df["factor_regional"].notna()].copy()
print(f"Filas tras borrado lógico IPC: {len(df)}")

# guardado de datos limpios 
os.makedirs("data", exist_ok=True)
df.to_csv(guardar_path, index=False)
print(f"\nDataset limpio guardado con éxito en: {guardar_path}")