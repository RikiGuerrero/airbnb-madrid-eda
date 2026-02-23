# Análisis Exploratorio de Datos: Airbnb Madrid

## 1) Objetivo
- Entender qué factores explican el **precio de los alojamientos en Airbnb Madrid** y cómo se relacionan con demanda (proxy: reseñas), disponibilidad y calidad (ratings).
- Comparar diferencias de precio por **tipo de alojamiento**, **barrio** y **estatus Superhost**.

## 2) Dataset
- **Fuente:** [Inside Airbnb – Madrid](http://insideairbnb.com/) (`listings.csv`)
- **Filas/columnas:** 25.000 filas × 75 columnas originales (reducidas a 13 variables clave tras selección).
- **Variables clave:**
  | Variable | Descripción |
  |---|---|
  | `price` | Precio por noche (€) |
  | `room_type` | Tipo de alojamiento (Entire home, Private room, etc.) |
  | `neighbourhood_cleansed` | Barrio |
  | `accommodates` | Nº de personas |
  | `bedrooms` / `beds` | Habitaciones y camas |
  | `host_is_superhost` | Estatus Superhost (t/f) |
  | `number_of_reviews` | Nº de reseñas (proxy de demanda) |
  | `review_scores_rating` | Calificación promedio |
  | `availability_365` | Disponibilidad anual (días) |
  | `amenities` | Lista de comodidades |
  | `property_type` | Tipo de propiedad |

## 3) Preguntas
- **Q1:** ¿Qué variables se asocian más con el precio de un alojamiento?
- **Q2:** ¿Existen diferencias significativas de precio entre barrios y tipos de alojamiento?
- **Q3:** ¿Ser Superhost se traduce en mayor precio o mayor demanda?

## 4) Data issues & fixes
| Problema | Solución |
|---|---|
| `price` en formato texto con `$` y comas | Parseo a `float64` en `clean_price()` |
| `amenities` como string JSON | Parseo a lista Python en `clean_amenities()` |
| `host_is_superhost` como texto (`t`/`f`) | Conversión a `bool` en `clean_boolean_columns()` |
| Columnas categóricas con espacios/inconsistencias | Normalización con `.str.strip()` + cast a `category` |
| Nulos en `bedrooms`, `beds` | Imputación por mediana agrupada (room_type × barrio) |
| Filas duplicadas | Eliminación con `drop_duplicates()` |
| 75 columnas originales (alta dimensionalidad) | Selección temprana a 13 variables con `BASE_KEEP_COLS` |

## 5) Pipeline
```
data/raw/listings.csv
  ↓  load_csv()
  ↓  selección de columnas (config.py → BASE_KEEP_COLS)
  ↓  clean_pipeline()
  │    ├── remove_duplicates
  │    ├── clean_boolean_columns
  │    ├── clean_price
  │    ├── clean_amenities
  │    ├── clean_neighbourhood
  │    ├── clean_categorical
  │    ├── impute_missing
  │    ├── drop_critical_nulls
  │    └── enforce_dtypes
  ↓  features_pipeline()
  │    ├── create_amenities_feature → amenities_count
  │    └── create_capacity_features → price_per_person
  ↓  visualizaciones + análisis estadístico
  ↓  save_csv() → data/processed/clean_dataset.csv
```

## 6) Hallazgos

### 📊 Insight 1: Factores más determinantes del precio
- **`accommodates`** es el driver principal del precio (mayor correlación positiva).
- Las variables de capacidad/infraestructura (`bedrooms`, `beds`) dominan el ranking.
- `amenities_count` tiene efecto positivo moderado.
- Las reseñas y ratings tienen relación débil/negativa con el precio.
- **→ Ver:** `p1_heatmap_correlacion.png`, `p1_ranking_correlacion_precio.png`

### 📊 Insight 2: Diferencias de precio por tipo y barrio
- **Entire home/apt** tiene la mediana de precio más alta; **Shared room** la más baja.
- Existe una diferencia significativa entre barrios (el más caro puede costar varias veces más que el más barato).
- Entire home es más **eficiente por persona** para grupos (menor `price_per_person`).
- La **combinación** ubicación + tipo genera la mayor dispersión de precios.
- **→ Ver:** `p2_boxplot_precio_room_type.png`, `p2_barras_precio_barrio.png`, `p2_heatmap_barrio_room_type.png`

### 📊 Insight 3: Impacto de ser Superhost
- Superhosts **NO cobran más** (mediana de precio similar a No-Superhosts).
- Superhosts reciben **más reseñas** → mayor volumen de reservas.
- Superhosts tienen **mejor rating**, consistente con los requisitos del programa.
- **→ Conclusión:** la estrategia Superhost es de **volumen**, no de premium.
- **→ Ver:** `p3_boxplot_superhost.png`, `p3_barras_superhost_diff.png`

## 7) Estructura del proyecto
```
Eda_project/
├── data/
│   ├── raw/
│   │   └── listings.csv              # Dataset original (Airbnb Madrid)
│   └── processed/
│       └── clean_dataset.csv         # Dataset limpio + features
├── notebooks/
│   └── eda.ipynb                     # Análisis exploratorio completo
├── src/
│   ├── __init__.py
│   ├── io_utils.py                   # Carga/guardado de datos
│   ├── cleaning.py                   # Pipeline de limpieza
│   ├── config.py                     # Rutas y columnas objetivo
│   ├── features.py                   # Creación de features
│   ├── viz.py                        # Visualizaciones reutilizables
│   └── utils.py                      # Validaciones y QC
├── visualizations/                   # Gráficos exportados (.png)
├── main.py                           # Pipeline end-to-end
├── requirements.txt                  # Dependencias
└── README.md                         # Este archivo
```

### Módulos clave:
- **`io_utils.py`**: `load_csv()`, `save_csv()`
- **`cleaning.py`**: `clean_pipeline()` — orquesta 9 funciones de limpieza
- **`features.py`**: `features_pipeline()` — crea `price_per_person` y `amenities_count`
- **`viz.py`**: gráficos reutilizables (heatmaps, boxplots, barras)
- **`utils.py`**: `qc_missing()`, `qc_duplicates()`
- **`config.py`**: `RAW_PATH`, `OUT_PATH`, `BASE_KEEP_COLS`

## 8) Cómo ejecutar

### Instalación de dependencias
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

### Ejecutar pipeline completo (end-to-end)
```bash
python main.py
```

### Ejecutar análisis exploratorio (notebook)
```bash
jupyter notebook notebooks/eda.ipynb
```

## 9) Conclusiones y próximos pasos

### ✅ Conclusión general
El precio en Airbnb Madrid está determinado **principalmente por factores estructurales** (tipo de alojamiento, capacidad, localización) más que por reputación. Ser Superhost atrae demanda pero no justifica un premium en tarifa. La estrategia ganadora es: **buena ubicación + infraestructura adecuada + calidad consistente**.

### 🚀 Posibles ampliaciones
1. **Modelado predictivo:** regresión del precio basada en las features creadas.
2. **Análisis temporal:** variación de precios/disponibilidad por estación.
3. **NLP:** análisis de sentimiento en descripciones de los listados.
4. **Recomendaciones:** precios óptimos por barrio/tipo para nuevos hosts.

---

**Proyecto EDA — Máster Data Science & IA, 2026**
