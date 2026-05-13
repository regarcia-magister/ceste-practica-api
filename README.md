# Practica Deploy ML

API para clasificar flores Iris con un modelo de Machine Learning desplegable en la nube.

## Estructura

- `train_model.py`: entrena y evalua varios modelos, genera la curva ROC y guarda `model.pkl`.
- `app.py`: expone la API con FastAPI.
- `requirements.txt`: dependencias del proyecto.
- `Dockerfile`: configuracion para despliegue en Render o Railway.
- `model.pkl`: modelo entrenado listo para inferencia.

## Modelo elegido

Se evaluaron varios algoritmos con validacion cruzada y control basico de sobreajuste:

- LogisticRegression
- RandomForestClassifier
- SVC
- KNeighborsClassifier

El modelo elegido para despliegue es `LogisticRegression` por su buen equilibrio entre rendimiento, estabilidad e interpretabilidad.

## Endpoints

- `GET /`: informacion general de la API.
- `GET /health`: estado del servicio.
- `POST /predict`: prediccion de especie Iris.

## Ejemplo de peticion

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

## Ejecucion local

```powershell
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe app.py
```

La API queda disponible en `http://127.0.0.1:8000`.
