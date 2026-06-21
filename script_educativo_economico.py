import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def mediana_ponderada(valores, pesos):
    orden = np.argsort(valores)
    valores_ord = np.array(valores)[orden]
    pesos_ord = np.array(pesos)[orden]
    pesos_acum = np.cumsum(pesos_ord) / pesos_ord.sum()
    return valores_ord[np.searchsorted(pesos_acum, 0.5)]

# carga del dataset
df = pd.read_csv(r"data\eph_formosa_ushuaia_limpio.csv")
df = df[(df["ESTADO_LIMPIO"] == 1) & (df["P21_REAL"] > 0)].copy()
df = df[df["NIVEL_ED_LIMPIO"].notna()].copy()

# mapeos
aglomerados = {
    15: "Formosa",
    31: "Ushuaia-Río Grande"
}

df["AGLOMERADO_NOMBRE"] = df["AGLOMERADO"].map(
    aglomerados
)

niveles = {
    1: "Primario incompleto",
    2: "Primario completo",
    3: "Secundario incompleto",
    4: "Secundario completo",
    5: "Superior incompleto",
    6: "Superior completo",
    7: "Sin instrucción"
}

df["NIVEL_ED_NOMBRE"] = (
    df["NIVEL_ED_LIMPIO"].map(niveles)
)

orden_niveles = ["Sin instrucción", "Primario incompleto",
                 "Primario completo", "Secundario incompleto",
                 "Secundario completo", "Superior incompleto", "Superior completo"]

df["NIVEL_ED_NOMBRE"] = pd.Categorical(df["NIVEL_ED_NOMBRE"], categories=orden_niveles, ordered=True)

# grafico 1: ingreso mediano real por nivel educativo
print("\nINGRESO MEDIANO REAL POR NIVEL EDUCATIVO\n")

ingresos_educacion = (
    df.groupby(["AGLOMERADO_NOMBRE","NIVEL_ED_NOMBRE"], observed=True
    )
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index(name="Ingreso Mediano") 
) 

plt.figure(figsize=(14, 6))
sns.barplot(data=ingresos_educacion, x="NIVEL_ED_NOMBRE", y="Ingreso Mediano", hue="AGLOMERADO_NOMBRE")
plt.xticks(rotation=45)
plt.title("Ingreso real mediano de la ocupación principal según nivel educativo")
plt.tight_layout()
plt.show()

# grafico 2: evolucion historica por nivel de educacion
df_evolucion = df[df["NIVEL_ED_NOMBRE"] != "Sin instrucción"].copy()
evolucion = (
    df_evolucion.groupby(["ANO4", "AGLOMERADO_NOMBRE", "NIVEL_ED_NOMBRE"])
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index(name="Ingreso Mediano")
)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
fig.suptitle("Evolución del ingreso real mediano por nivel educativo (2016-2025)")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia-Río Grande"]):
    datos = evolucion[evolucion["AGLOMERADO_NOMBRE"] == aglo]
    sns.lineplot(data=datos, x="ANO4", y="Ingreso Mediano", hue="NIVEL_ED_NOMBRE", marker="o", ax=ax)
    ax.set_title(aglo)
    ax.grid(True)
plt.tight_layout()
plt.show()

# clasificacion caes para ramas de actividades
df["PP04B_2DIG"] = (df["PP04B_COD_LIMPIO"] // 100).astype("Int64")

# mejora visual para los graficos, para que queden nombres especificos y no códigos ilegibles
caes_original = pd.read_excel("caes_v2018.xls", engine="xlrd", header=2) # cargamos el archivo caes del indec (clasificador de actividades económicas)
caes_original.columns = ["codigo", "descripcion"] # renombramos las columnas para facilitar trabajo
caes_original = caes_original.dropna(subset=["codigo"]) # eliminamos filas sin codigo
caes_original["codigo"] = caes_original["codigo"].astype(str).str.strip() # convertimos codigos a string y eliminamos espacios
caes_filtrado = caes_original[caes_original["codigo"].str.match(r"^\d{2}$")].copy() # filtramos codigos de dos digitos (por rubro)
caes_filtrado["codigo_int"] = caes_filtrado["codigo"].astype(int)  # convertimos a entero para que coincida con los valores de PP04B_2DIG
caes_labels = dict(zip(caes_filtrado["codigo_int"], caes_filtrado["descripcion"].str.strip().str.title()))  # creamos diccionario {codigo : descripcion} para mapear
df["sector_nombre"] = df["PP04B_2DIG"].map(caes_labels) # mapeamos codigo de 2 digitos al nombre del sector

# grafico 3: participacion por sector
sectores = (
    df.groupby(["AGLOMERADO_NOMBRE", "sector_nombre"])["PONDERA"].sum().reset_index()
)
sectores["Participacion (%)"] = sectores.groupby("AGLOMERADO_NOMBRE")["PONDERA"].transform(lambda x: x / x.sum() * 100)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Top 10 Sectores Económicos con Mayor Participación de Ocupados")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia-Río Grande"]):
    top10 = sectores[sectores["AGLOMERADO_NOMBRE"] == aglo].sort_values("Participacion (%)", ascending=False).head(10)
    sns.barplot(data=top10, x="Participacion (%)", y="sector_nombre", ax=ax)
    ax.set_title(aglo)
plt.tight_layout()
plt.show()

# grafico 4: sectores con mayores ingresos
ingresos_sector = (
    df.groupby(["AGLOMERADO_NOMBRE", "sector_nombre"])
    .apply(lambda x: mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values), include_groups=False)
    .reset_index(name="Ingreso Mediano"))

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
fig.suptitle("Top 10 Sectores Económicos con Mayores Ingresos Medianos")

for ax, aglo in zip(axes, ["Formosa", "Ushuaia-Río Grande"]):
    top10 = ingresos_sector[ingresos_sector["AGLOMERADO_NOMBRE"] == aglo].sort_values("Ingreso Mediano", ascending=False).head(10)
    top10 = top10.copy()
    top10["sector_corto"] = top10["sector_nombre"].str[:40]
    sns.barplot(data=top10, x="Ingreso Mediano", y="sector_corto", ax=ax)
    ax.set_title(aglo)
    ax.set_ylabel("")
plt.tight_layout()
plt.show()

# tabla resumen
tabla_resumen = (
    df.groupby(["AGLOMERADO_NOMBRE", "NIVEL_ED_NOMBRE"], observed=True)
    .apply(lambda x: pd.Series({
        "media": np.average(x["P21_REAL"], weights=x["PONDIIO"]),
        "mediana": mediana_ponderada(x["P21_REAL"].values, x["PONDIIO"].values)
    }), include_groups=False)
    .round(2)
)

print("\nTABLA RESUMEN\n")
print(tabla_resumen)