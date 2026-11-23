# Predictor de votos en el Congreso

Modelo de machine learning que estima cómo votarían los diputados nacionales argentinos un proyecto en tratamiento. Se pega el título o resumen del proyecto, Gemini lo clasifica y un Random Forest calcula la probabilidad de voto afirmativo de cada diputado en ejercicio.

Trabajo final de Ciencia de Datos para Politólogos, Carrera de Ciencia Política, Universidad de Buenos Aires.

## Datos

| Dato | Valor |
| --- | --- |
| Votos | 764.389 |
| Votaciones | 2.992 |
| Diputados | 2.070 |
| Período | junio de 1993 a octubre de 2026 |

Fuentes:

- Votaciones nominales del sitio oficial de la Cámara de Diputados: [votaciones.hcdn.gob.ar](https://votaciones.hcdn.gob.ar).
- `Datos/diputados.json`, del proyecto [Como Voto](https://github.com/rquiroga7/Como_voto), con las votaciones hasta abril de 2026.
- `Datos/votos_nuevos.csv`, descargado con `scraper.ipynb`, con las votaciones de mayo a octubre de 2026.
- `Datos/clasificacion_bloques.csv`, clasificación propia de 307 nombres de bloque en 7 macrocoaliciones: Peronismo, Radicalismo, PRO, LLA, Provinciales, Izquierda y Otros.

## Estructura

| Archivo | Qué hace |
| --- | --- |
| `scraper.ipynb` | Descarga del sitio oficial las votaciones posteriores al JSON |
| `dataset.ipynb` | Expande y limpia los votos, asigna macrocoaliciones y calcula las variables |
| `modelo.ipynb` | Entrena y evalúa el Random Forest y guarda `modelo_random_forest.pkl` |
| `Agente/agente.ipynb` | Recibe un proyecto, lo clasifica con Gemini y predice el voto de cada diputado |
| `modelo_random_forest.pkl` | Modelo entrenado |
| `Datos/` | Datos de entrada |

## Cómo correrlo

1. Instalar las librerías: `pip install -r requirements.txt`.
2. Correr `scraper.ipynb` (opcional: `votos_nuevos.csv` ya está incluido).
3. Correr `dataset.ipynb`. Genera en `Datos/` los CSV intermedios y `dataset_features.csv`. No están en el repositorio porque pesan más de 100 MB cada uno.
4. Correr `modelo.ipynb`.
5. Correr `Agente/agente.ipynb`. Pide una clave de Gemini, que se crea en [Google AI Studio](https://aistudio.google.com/api-keys). Se ingresa oculta con `getpass` y no queda guardada en el notebook.

## Modelo

Clasificación binaria: el target es `es_afirmativo` (1 si el diputado votó afirmativo, 0 si votó en contra, se abstuvo o faltó).

Variables:

- Del diputado: macrocoalición, provincia, si es oficialista, tasas de afirmativos, negativos, abstenciones y ausencias en toda su carrera, tasas de afirmativos y ausencias en sus últimos 50 votos, y cantidad de votos previos.
- De la votación: quién presentó el proyecto (Poder Ejecutivo, Senado, Diputados o Jefatura de Gabinete), si es una moción, año y mes.

Las tasas usan solo votos anteriores a cada votación, para evitar data leakage. La separación entre entrenamiento, validación y test es temporal: 70% de las votaciones más antiguas, 15% siguiente y 15% más reciente.

Random Forest de 100 árboles, profundidad máxima 12 y clases balanceadas.

## Resultados

| Conjunto | Período | Accuracy | F1 | Línea base |
| --- | --- | --- | --- | --- |
| Validación | marzo de 2018 a julio de 2023 | 66,4% | 0,737 | 62,1% |
| Test | julio de 2023 a octubre de 2026 | 70,2% | 0,741 | 57,3% |

La línea base es el acierto de predecir siempre la clase más frecuente. En test, el modelo la supera por 12,9 puntos.

## Limitaciones

- El modelo no conoce el tema del proyecto: de la ley solo usa quién la presentó y si es una moción. La predicción depende sobre todo del historial de cada diputado y de si es oficialista.
- Aprende de votaciones pasadas: no anticipa cambios de postura por negociaciones o hechos nuevos.
- Una ausencia cuenta igual que un voto negativo.
- La clasificación de bloques en macrocoaliciones y la definición de oficialismo son decisiones del grupo.
- La mitad de los títulos de votación no incluye número de expediente, así que su origen queda sin dato.

## Uso

Proyecto académico. Los datos de votaciones provienen de fuentes públicas; el JSON histórico y la lógica del scraper se basan en el proyecto Como Voto.
