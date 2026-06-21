import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/eph_formosa_ushuaia_limpio.csv")

df["aglomerado"] = df["AGLOMERADO"].map({15: "Formosa", 31: "Ushuaia"})

def mediana_ponderada(valores, pesos):
    orden = np.argsort(valores)
    valores_ord = np.array(valores)[orden]
    pesos_ord = np.array(pesos)[orden]
    pesos_acum = np.cumsum(pesos_ord) / pesos_ord.sum()
    return valores_ord[np.searchsorted(pesos_acum, 0.5)]

# calculo de tasas
# Las tasas se calculan con pondera 
# estados:
# 1 = Ocupado
# 2 = Desocupado
# 3 = Inactivo
# 0 = Menor de 10 años (no integra la PEA)
# PEA = ocupados + desocupados
# Tasa de actividad = PEA / población total (excluye menores de 10)
# Tasa de empleo = ocupados / población total (excluye menores de 10)
# Tasa de desocupación = desocupados / PEA

# para excluir a menores de 10 años del denominador de actividad y empleo
df_mayor10 = df[df["ESTADO_LIMPIO"] != 0].copy()

def calcular_tasas(grupo):
    pond_total = grupo["PONDERA"].sum()
    pond_ocup = grupo[grupo["ESTADO_LIMPIO"] == 1]["PONDERA"].sum()
    pond_desoc = grupo[grupo["ESTADO_LIMPIO"] == 2]["PONDERA"].sum()
    pea = pond_ocup + pond_desoc

    return pd.Series({
        "tasa_actividad": 100 * pea / pond_total if pond_total > 0 else np.nan,
        "tasa_empleo": 100 * pond_ocup / pond_total if pond_total > 0 else np.nan,
        "tasa_desocupacion": 100 * pond_desoc / pea if pea > 0 else np.nan,
    })

tasas = (
    df_mayor10.groupby(["ANO4", "TRIMESTRE", "aglomerado"])
    .apply(calcular_tasas, include_groups=False)
    .reset_index()
)

tasas["periodo"] = tasas["ANO4"].astype(str) + "-T" + tasas["TRIMESTRE"].astype(str)

# ingresos reales (solo ocupados con ingreso > 0)
df_ing = df[(df["ESTADO_LIMPIO"] == 1) & (df["P21_REAL"] > 0)].copy()

ingresos = (
    df_ing.groupby(["ANO4", "aglomerado"])
    .apply(lambda x: pd.Series({
        "ingreso_medio":   np.average(x["P21_REAL"], weights=x["PONDIIO"]),
        "ingreso_mediana": mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values)
    }), include_groups=False)
    .reset_index()
)

# grafico 1 (tasa de actividad)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Tasa de actividad (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    d = tasas[tasas["aglomerado"] == aglo]
    ax.plot(range(len(d)), d["tasa_actividad"], marker="o", color="steelblue")
    ax.set_title(aglo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Tasa (%)")
    ax.set_xticks(range(len(d["periodo"])))
    ax.set_xticklabels(d["periodo"], rotation=90, fontsize=7)
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=90, fontsize=7)

plt.tight_layout()
plt.show()

# grafico 2 (Tasa de empleo)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Tasa de empleo (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    d = tasas[tasas["aglomerado"] == aglo]
    ax.plot(range(len(d)), d["tasa_empleo"], marker="o", color="seagreen")
    ax.set_title(aglo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Tasa (%)")
    ax.set_xticks(range(len(d["periodo"])))
    ax.set_xticklabels(d["periodo"], rotation=90, fontsize=7)
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=90, fontsize=7)

plt.tight_layout()
plt.show()

# grafico 3 (tasa de desocupación)

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Tasa de desocupación (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    d = tasas[tasas["aglomerado"] == aglo]
    ax.plot(range(len(d)), d["tasa_desocupacion"], marker="o", color="tomato")
    ax.set_title(aglo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Tasa (%)")
    ax.set_xticks(range(len(d["periodo"])))
    ax.set_xticklabels(d["periodo"], rotation=90, fontsize=7)
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=7)

plt.tight_layout()
plt.show()

# grafico 4 (ingresos reales (media y mediana))

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig.suptitle("Ingresos reales ($ dic-2016) — ocupacion principal con ingreso > 0")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia"]):
    d = ingresos[ingresos["aglomerado"] == aglo]
    ax.plot(d["ANO4"], d["ingreso_medio"],   marker="o", label="Media ponderada")
    ax.plot(d["ANO4"], d["ingreso_mediana"], marker="s", linestyle="--", label="Mediana")
    ax.set_title(aglo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Ingreso real ($ dic-2016)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=90, fontsize=7)

plt.tight_layout()
plt.show()

# tabla resumen

resumen = tasas.groupby(["ANO4", "aglomerado"])[["tasa_actividad","tasa_empleo","tasa_desocupacion"]].mean().reset_index().merge(ingresos, on=["ANO4", "aglomerado"])
resumen = resumen.round(2)

print("\nTABLA RESUMEN — Tasas e ingresos por año y aglomerado\n")
print(resumen.to_string(index=False))
