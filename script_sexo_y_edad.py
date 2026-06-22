import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CARGA DEL DATASET

df = pd.read_csv("data/eph_formosa_ushuaia_limpio.csv")

print("Dataset cargado correctamente")
print(f"Cantidad de registros: {len(df)}")

# ETIQUETAS

df["sexo"] = df["CH04_LIMPIO"].map({
    1: "Varón",
    2: "Mujer"
})

df["aglomerado"] = df["AGLOMERADO"].map({
    15: "Formosa",
    31: "Ushuaia"
})

# GRUPOS DE EDAD

bins = [14, 24, 44, 64, 120]
labels = ["14-24", "25-44", "45-64", "65+"]

df["grupo_edad"] = pd.cut(
    df["CH06_LIMPIO"],
    bins=bins,
    labels=labels
)

def mediana_ponderada(valores, pesos):
    orden = np.argsort(valores)
    valores_ord = np.array(valores)[orden]
    pesos_ord = np.array(pesos)[orden]
    pesos_acum = np.cumsum(pesos_ord) / pesos_ord.sum()
    return valores_ord[np.searchsorted(pesos_acum, 0.5)]

# INGRESO REAL PROMEDIO POR SEXO

ingresos_sexo = (
    df[(df["ESTADO_LIMPIO"] == 1) & (df["P21_REAL"] > 0)]
    .groupby(["ANO4", "aglomerado", "sexo"], observed=True)
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index().rename(columns={0: "P21_REAL"})
    
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Evolución del ingreso real mediano por sexo (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    datos_aglo = ingresos_sexo[ingresos_sexo["aglomerado"] == aglo]
    for sexo in ["Varón", "Mujer"]:
        datos = datos_aglo[datos_aglo["sexo"] == sexo]
        ax.plot(datos["ANO4"], datos["P21_REAL"], marker="o", label=sexo)

    ax.set_title(f"Aglomerado {aglo}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Ingreso real mediano ($)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# INGRESO REAL PROMEDIO POR EDAD

ingresos_edad = (
    df[(df["ESTADO_LIMPIO"] == 1) & (df["P21_REAL"] > 0)]
    .groupby(["ANO4", "aglomerado", "grupo_edad"], observed=True)
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index().rename(columns={0: "P21_REAL"})
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("evolucion del ingreso real mediano por grupo etario (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    datos_aglo = ingresos_edad[ingresos_edad["aglomerado"] == aglo]
    for grupo in labels:
        datos = datos_aglo[datos_aglo["grupo_edad"] == grupo]
        ax.plot(datos["ANO4"], datos["P21_REAL"], marker="o", label=f"Edad: {grupo}")
    
    ax.set_title(f"Aglomerado {aglo}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Ingreso real mediano ($)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# TABLA RESUMEN

df_tabla = df[(df["ESTADO_LIMPIO"] == 1) & (df["P21_REAL"] > 0)].reset_index(drop=True)
tabla = (
    df_tabla.groupby(["aglomerado", "sexo"], observed=True)
    .apply(lambda x: pd.Series({
        "media_ponderada": np.average(x["P21_REAL"], weights=x["PONDIIO"]),
        "mediana_ponderada": mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values),
        "count": len(x)
    }), include_groups=False)
    .round(2)
)

print("TABLA RESUMEN (2016-2025)")
print(tabla)