import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

df = pd.read_csv("data/eph_formosa_ushuaia_limpio.csv")

# variables independientes disponibles
df["sexo"] = df["CH04_LIMPIO"]
df["edad"] = df["CH06_LIMPIO"]
df["nivel_ed"] = df["NIVEL_ED_LIMPIO"]
df["aglomerado"] = df["AGLOMERADO"]
df["PP04B_2DIG"] = (df["PP04B_COD_LIMPIO"] // 100).astype("Int64")

# Solo ocupados (ESTADO == 1)
df_ocupados = df[df["ESTADO_LIMPIO"] == 1].copy()

# =============================================================================
# transformacion de variables categoricas a numeros (0 y 1) (dummificar)
# =============================================================================
FEATURES = ["sexo", "edad", "nivel_ed", "aglomerado", "PP04B_2DIG", "ANO4"]

# pasamos las nominales a string para que get_dummies las procese como categoricas
categoricas = ["sexo", "nivel_ed", "aglomerado", "PP04B_2DIG", "ANO4"]
for col in categoricas:
    df_ocupados[col] = df_ocupados[col].astype(str)

# se convierten a string las categoricas y dummificamos todo en un solo paso
X_dummies = pd.get_dummies(df_ocupados[FEATURES], columns=categoricas, drop_first=True)
X_dummies["edad"] = df_ocupados["edad"].astype(float)

# columnas finales del modelo lineal
columnas_modelo = X_dummies.columns.tolist()

# pegamos target y ponderadores juntos
X_dummies["P21_REAL"] = df_ocupados["P21_REAL"]
X_dummies["P21_LIMPIO"] = df_ocupados["P21_LIMPIO"]
X_dummies["PONDIIO"] = df_ocupados["PONDIIO"]

# =============================================================================
# SEPARACION DE MUESTRAS
# =============================================================================

# con ingreso conocido y datos completos (para entrenar)
df_conocido = X_dummies[
    X_dummies["P21_LIMPIO"].notna() &
    X_dummies["P21_REAL"].notna() &
    (X_dummies["P21_REAL"] > 0)
].dropna(subset=columnas_modelo).copy()

# sin ingreso / a imputar
df_no_respuesta = X_dummies[
    X_dummies["P21_LIMPIO"].isna()
].dropna(subset=columnas_modelo).copy()

print(f"Casos con ingreso conocido: {len(df_conocido)}")
print(f"Casos sin ingreso (no respuesta): {len(df_no_respuesta)}")

X_final = df_conocido[columnas_modelo].astype(float)
y_final = np.log1p(df_conocido["P21_REAL"])
pesos_final = df_conocido["PONDIIO"]

# =============================================================================
# ENTRENAMIENTO Y EVALUACION PONDERADA
# =============================================================================
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_final, y_final, pesos_final, test_size=0.2, random_state=42
)

modelo = LinearRegression()
modelo.fit(X_train, y_train, sample_weight=w_train)

y_pred = modelo.predict(X_test)

# metricas de evaluación ponderadas
r2 = r2_score(y_test, y_pred, sample_weight=w_test)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred, sample_weight=w_test))

print(f"\nR² (Ponderado):   {r2:.4f}")
print(f"RMSE (Escala log): {rmse_log:.4f}")
print(f"RMSE ($ dic-2016):  ${np.expm1(rmse_log):.2f}") # conversión a pesos reales

# =============================================================================
# INTERPRETACION DE COEFICIENTES (FILTRADO PARA LEGIBILIDAD)
# =============================================================================
coeficientes = pd.Series(modelo.coef_, index=columnas_modelo).sort_values()

print("\nCoeficientes del modelo (Top 15 mayores impactos sobre log(ingreso)):")
print(coeficientes.tail(15).round(4))

# seleccionamos los 10 extremos de cada lado para evitar que se encimen las etiquetas (estetico / legibilidad)
top_positivos = coeficientes.tail(10)
top_negativos = coeficientes.head(10)
coefs_filtrados = pd.concat([top_negativos, top_positivos])

plt.figure(figsize=(13, 8))
colores = np.where(coefs_filtrados >= 0, "steelblue", "crimson")

# mejora visual para los graficos, para que queden nombres especificos y no códigos ilegibles
caes_original = pd.read_excel("caes_v2018.xls", engine="xlrd", header=2) # cargamos el archivo caes del indec (clasificador de actividades económicas)
caes_original.columns = ["codigo", "descripcion"] # renombramos las columnas para facilitar trabajo
caes_original = caes_original.dropna(subset=["codigo"]) # eliminamos filas sin codigo
caes_original["codigo"] = caes_original["codigo"].astype(str).str.strip() # convertimos codigos a string y eliminamos espacios
caes_filtrado = caes_original[caes_original["codigo"].str.match(r"^\d{2}$")].copy() # filtramos codigos de dos digitos (por rubro)
caes_filtrado["codigo_int"] = caes_filtrado["codigo"].astype(int) # convertimos a entero para que coincida con los valores de PP04B_2DIG
caes_labels = dict(zip(caes_filtrado["codigo_int"], caes_filtrado["descripcion"].str.strip().str.title())) # creamos diccionario {codigo : descripcion} para mapear
df["sector_nombre"] = df["PP04B_2DIG"].map(caes_labels) # mapeamos codigo de 2 digitos al nombre del sector

# renombrar indice del gráfico
def renombrar(nombre):
    if "PP04B_2DIG_" in nombre:
        cod = int(nombre.replace("PP04B_2DIG_", ""))
        return caes_labels.get(cod, nombre)
    return nombre

coefs_filtrados.index = [renombrar(n) for n in coefs_filtrados.index]

coefs_filtrados.plot(kind="barh", color=colores)
plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
plt.title("top 20 variables con mayor impacto sobre el ingreso real (log)", fontsize=12)
plt.xlabel("coeficiente (Impacto relativo)")
plt.ylabel("variables dummificadas")
plt.grid(axis="x", linestyle=":", alpha=0.6)
plt.subplots_adjust(left=0.35)
plt.show()

# =============================================================================
# IMPUTACIÓN DE NO RESPUESTA
# =============================================================================
X_imputar = df_no_respuesta[columnas_modelo].astype(float)

if len(X_imputar) > 0:
    y_imputado_log = modelo.predict(X_imputar)
    y_imputado = np.expm1(y_imputado_log)

    print(f"\nimputación completada para {len(X_imputar)} casos")
    print(f"ingreso imputado promedio: ${y_imputado.mean():.2f}")
    print(f"ingreso imputado mediana:  ${np.median(y_imputado):.2f}")
else:
    print("\nno hay casos a imputar con variables completas.")

# =============================================================================
# GRAFICO: VALORES REALES VS PREDICHOS
# =============================================================================
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color="steelblue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("log(ingreso real)")
plt.ylabel("log(ingreso predicho)")
plt.title(f"real vs predicho (ponderado) — R²={r2:.3f}")
plt.tight_layout()
plt.show()

# =============================================================================
# ANALISIS DE RESIDUOS Y SUPUESTOS
# =============================================================================
residuos = y_test - y_pred

# QQ plot 
fig, ax = plt.subplots(figsize=(8, 5))
sm.qqplot(residuos, line='45', fit=True, ax=ax)
plt.title("QQ Plot de Residuos (Verificación de Normalidad)")
plt.tight_layout()
plt.show()

# residuos vs predichos 
plt.figure(figsize=(8, 5))
plt.scatter(y_pred, residuos, alpha=0.3, color="steelblue")
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valores predichos")
plt.ylabel("Residuos")
plt.title("Residuos vs Predicciones (Análisis de Homocedasticidad)")
plt.tight_layout()
plt.show()