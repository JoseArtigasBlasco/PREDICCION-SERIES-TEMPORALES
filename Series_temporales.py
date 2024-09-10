# Series temporales en Python utilizando la biblioteca Pandas para el análisis
# y la visualización de datos. Statsmodels para modelar y predecir series temporales.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Generar datos de ejemplo
np.random.seed(123)
fecha_inicio = '2020-01-01'
fecha_fin = '2021-12-31'
rango_fechas = pd.date_range(start=fecha_inicio, end=fecha_fin)
datos = np.random.randn(len(rango_fechas))
serie_temporal = pd.Series(datos, index=rango_fechas)

# Visualización de la serie temporal
plt.figure(figsize=(10, 6))
plt.plot(serie_temporal)
plt.title('Serie Temporal de Ejemplo')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.grid(True)
plt.show()

# # Descompone la serie temporal en tendencia, estacionalidad y residuo.
descomposicion = sm.tsa.seasonal_decompose(serie_temporal, model='additive')
tendencia = descomposicion.trend
estacionalidad = descomposicion.seasonal
residuo = descomposicion.resid

# Visualización de la descomposición
plt.figure(figsize=(10, 8))

plt.subplot(4,1,1)
plt.plot(serie_temporal, label='Serie Temporal')
plt.legend(loc='upper left')

plt.subplot(4,1,2)
plt.plot(tendencia, label='Tendencia')
plt.legend(loc='upper left')

plt.subplot(4,1,3)
plt.plot(estacionalidad, label='Estacionalidad')
plt.legend(loc='upper left')

plt.subplot(4,1,4)
plt.plot(residuo, label='Residuo')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()


# Ajusta un modelo ARIMA a los datos.
modelo = sm.tsa.ARIMA(serie_temporal, order=(1, 1, 1))  # Orden (p, d, q)
resultados = modelo.fit(method_kwargs={"disp": -1})

# Realiza una predicción futura utilizando el modelo ARIMA.
prediccion_inicio = '2022-01-01'
prediccion_fin = '2022-06-30'
prediccion = resultados.predict(start=prediccion_inicio, end=prediccion_fin, dynamic=False)

# Visualización de la predicción
plt.figure(figsize=(10, 6))
plt.plot(serie_temporal, label='Datos Observados')
plt.plot(prediccion, label='Predicción', color='red')
plt.title('Predicción de la Serie Temporal')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.legend()
plt.grid(True)
plt.show()










