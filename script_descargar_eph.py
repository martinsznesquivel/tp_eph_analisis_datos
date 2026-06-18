import os
import time
import pyeph

OUTPUT_DIR = "./datos_indec"
os.makedirs(OUTPUT_DIR, exist_ok=True) # si no existe se crea

# EPH empezó a publicar desde el 2 trimestre de 2016, osea, son 39 en total y no 40.
periodos = []
for año in range(2016, 2026):
    for trimestre in range(1, 5):
        if año == 2016 and trimestre == 1:
            continue  # como dice arriba, el primer cuatrimestre de 2016 no está publicado, sin esto rompería siempre
        periodos.append((año, trimestre))

print(f"Se van a intentar descargar {len(periodos)} trimestres: {periodos[0]} a {periodos[-1]}")

errores = []

for año, trimestre in periodos:
    ind_path = os.path.join(OUTPUT_DIR, f"individual_{año}_T{trimestre}.csv")

    # Si ya está descargado se saltea
    if os.path.exists(ind_path):
        print(f"{año} T{trimestre}: ya descargado, salteo.")
        continue

    print(f"-> Descargando {año} T{trimestre} ...", end=" ")
    try:
        individual = pyeph.obtener(data="eph", ano=año, periodo=trimestre, tipo_base="individual")

        individual.to_csv(ind_path, index=False)

        print(f"Ok ({len(individual)} personas)")

    except Exception as error:
        print(f"Error: {error}")
        errores.append((año, trimestre, str(error)))

    time.sleep(1)  # para no saturar al indec y que no nos termine bloqueando las solicitudes por ip

print("\nDescarga finalizada")
if errores:
    print(f"{len(errores)} trimestres fallaron:")
    for año, trimestre, mensaje in errores:
        print(f"  - {año} T{trimestre}: {mensaje}")
    print("\nPodés volver a correr el script: los trimestres ya descargados se saltean")
    print("y solo va a reintentar los que fallaron.")
else:
    print("Todos los trimestres se descargaron correctamente.")