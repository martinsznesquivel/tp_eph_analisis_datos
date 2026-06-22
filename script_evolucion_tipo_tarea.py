import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def mediana_ponderada(valores, pesos):
    orden = np.argsort(valores)
    valores_ord = np.array(valores)[orden]
    pesos_ord = np.array(pesos)[orden]
    pesos_acum = np.cumsum(pesos_ord) / pesos_ord.sum()
    return valores_ord[np.searchsorted(pesos_acum, 0.5)]

# carga del dataset
df = pd.read_csv("data/eph_formosa_ushuaia_limpio.csv")
print("Dataset cargado correctamente")
print(df.columns.tolist())
print(f"Cantidad de registros: {len(df)}")

# mapeos
df["aglomerado"] = df["AGLOMERADO"].map({
    15: "Formosa",
    31: "Ushuaia"
})

# calificacion de la tarea segun CNO-2001 (ultimo digito de PP04D_COD)
# 1 = Profesional  2 = Tecnica  3 = Operativa  4 = No calificada
ocupados = df[df["ESTADO_LIMPIO"] == 1].copy()
ocupados = ocupados[ocupados["PP04D_COD_LIMPIO"].notna()].copy()

ocupados["ultimo_digito"] = ocupados["PP04D_COD_LIMPIO"].astype("int64") % 10
mapa_calificacion = {1: "Profesional", 2: "Tecnica", 3: "Operativa", 4: "No calificada"}
ocupados["tipo_tarea"] = ocupados["ultimo_digito"].map(mapa_calificacion)

orden_tarea = ["No calificada", "Operativa", "Tecnica", "Profesional"]
df_tarea = ocupados[ocupados["tipo_tarea"].notna()].copy()
df_tarea["tipo_tarea"] = pd.Categorical(df_tarea["tipo_tarea"], categories=orden_tarea, ordered=True)

# composicion por tipo de tarea (participacion % sobre ocupados)
comp = (
    df_tarea.groupby(["ANO4", "aglomerado", "tipo_tarea"], observed=True)["PONDERA"]
    .sum()
    .reset_index()
)
comp["participacion"] = comp.groupby(["ANO4", "aglomerado"])["PONDERA"].transform(
    lambda x: 100 * x / x.sum()
)

# grafico 1: evolucion de la composicion ocupacional por tipo de tarea
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Evolución de la composición ocupacional por tipo de tarea (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    datos_aglo = comp[comp["aglomerado"] == aglo]
    for tarea in orden_tarea:
        datos = datos_aglo[datos_aglo["tipo_tarea"] == tarea]
        ax.plot(datos["ANO4"], datos["participacion"], marker="o", label=tarea)
    ax.set_title(f"Aglomerado {aglo}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Participación sobre ocupados (%)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# grafico 2: evolucion del ingreso real mediano por tipo de tarea
df_ing = df_tarea[df_tarea["P21_REAL"] > 0].copy()

ingresos_tarea = (
    df_ing.groupby(["ANO4", "aglomerado", "tipo_tarea"], observed=True)
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index()
    .rename(columns={0: "P21_REAL"})
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Evolución del ingreso real mediano por tipo de tarea (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    datos_aglo = ingresos_tarea[ingresos_tarea["aglomerado"] == aglo]
    for tarea in orden_tarea:
        datos = datos_aglo[datos_aglo["tipo_tarea"] == tarea]
        ax.plot(datos["ANO4"], datos["P21_REAL"], marker="o", label=tarea)
    ax.set_title(f"Aglomerado {aglo}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Ingreso real mediano ($ dic-2016)")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# grafico 3: composicion 2016 vs 2025
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Composición por tipo de tarea: 2016 vs. 2025")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    datos_aglo = comp[comp["aglomerado"] == aglo]
    extremos = datos_aglo[datos_aglo["ANO4"].isin([2016, 2025])]
    tabla_pivot = extremos.pivot(index="tipo_tarea", columns="ANO4", values="participacion")
    tabla_pivot = tabla_pivot.reindex(orden_tarea)
    tabla_pivot.plot(kind="barh", ax=ax)
    ax.set_title(f"Aglomerado {aglo}")
    ax.set_xlabel("Participación sobre ocupados (%)")
    ax.set_ylabel("")
    ax.legend(title="Año")
    ax.grid(True, axis="x")

plt.tight_layout()
plt.show()

# tabla resumen
df_tabla = df_ing.reset_index(drop=True)
tabla = (
    df_tabla.groupby(["aglomerado", "tipo_tarea"], observed=True)
    .apply(lambda x: pd.Series({
        "media_ponderada": np.average(x["P21_REAL"], weights=x["PONDIIO"]),
        "mediana_ponderada": mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values),
        "count": len(x)
    }), include_groups=False)
    .round(2)
)

print("\nTABLA RESUMEN (2016-2025)")
print(tabla)