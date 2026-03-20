# Análisis Exploratorio de Datos: Airbnb Madrid

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-3.0-green.svg)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 1) Objetivo

Analizar los factores que determinan el **precio de los alojamientos en Airbnb Madrid** y su relación con demanda, disponibilidad y calidad.

**Preguntas de investigación:**
- ¿Qué variables se asocian más con el precio?
- ¿Existen diferencias significativas entre barrios y tipos de alojamiento?
- ¿Ser Superhost se traduce en mayor precio o mayor demanda?

## 2) Dataset

| Atributo | Valor |
|----------|-------|
| **Fuente** | [Inside Airbnb – Madrid](http://insideairbnb.com/) |
| **Archivo** | `listings.csv` |
| **Registros** | 25,000 alojamientos |
| **Columnas originales** | 79 |
| **Columnas procesadas** | 26 (tras feature engineering) |

### Variables principales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `price` | float | Precio por noche (€) |
| `room_type` | category | Entire home, Private room, Shared room, Hotel room |
| `neighbourhood_cleansed` | category | Barrio (128 únicos) |
| `accommodates` | int | Capacidad de personas |
| `bedrooms` / `beds` | float | Habitaciones y camas |
| `host_is_superhost` | bool | Estatus Superhost |
| `number_of_reviews` | int | Nº de reseñas (proxy de demanda) |
| `review_scores_rating` | float | Calificación promedio (0-5) |
| `availability_365` | int | Días disponibles al año |

### Features creadas

| Feature | Descripción |
|---------|-------------|
| `amenities_count` | Número de comodidades del alojamiento |
| `price_per_person` | Precio por persona (price / accommodates) |
| `price_bucket` | Segmento de precio: budget / standard / premium / luxury |
| `occupancy_ratio` | Ratio de ocupación estimado (0-1) |
| `rating_bucket` | Categoría de rating: sin_rating / bajo / medio / alto / excelente |
| `host_type` | Tipo de anfitrión: individual / small_host / professional |

## 3) Pipeline de Procesamiento

```
┌─────────────────────────────────────────────────────────────────┐
│  data/raw/listings.csv (25,000 × 79)                            │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. SELECCIÓN DE COLUMNAS                                       │
│     get_keep_cols() → 21 columnas (base + opcionales)           │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. LIMPIEZA (clean_pipeline)                                   │
│     ├── remove_duplicates()                                     │
│     ├── clean_boolean_columns()    t/f → True/False             │
│     ├── clean_price()              "$1,234" → 1234.0            │
│     ├── clean_amenities()          JSON → lista Python          │
│     ├── clean_neighbourhood()      normalización + category     │
│     ├── clean_categorical()        property_type, room_type     │
│     ├── impute_missing()           mediana por grupos           │
│     ├── drop_critical_nulls()      elimina filas sin precio     │
│     └── enforce_dtypes()           esquema de tipos final       │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. VALIDACIÓN POST-LIMPIEZA (run_validation_suite)             │
│     ├── validate_nulls()           check nulos críticos         │
│     ├── validate_duplicates()      check duplicados             │
│     ├── validate_distributions()   comparación antes/después    │
│     ├── validate_dtypes()          verificación de tipos        │
│     └── validate_cardinality()     cardinalidad categóricas     │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. FEATURE ENGINEERING (features_pipeline)                     │
│     ├── create_amenities_feature()  → amenities_count           │
│     ├── create_capacity_features()  → price_per_person          │
│     ├── create_price_bucket()       → price_bucket              │
│     ├── create_occupancy_ratio()    → occupancy_ratio           │
│     ├── create_rating_bucket()      → rating_bucket             │
│     └── create_host_type()          → host_type                 │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. ANÁLISIS (run_analytics_suite)                              │
│     ├── summary_by_room_type()                                  │
│     ├── summary_by_neighbourhood()                              │
│     ├── summary_by_price_bucket()                               │
│     ├── summary_superhost_comparison()                          │
│     ├── pivot_price_by_neighbourhood_roomtype()                 │
│     ├── summary_by_host_type()                                  │
│     └── summary_correlations()                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. VISUALIZACIONES + EXPORTACIÓN                               │
│     → data/processed/clean_dataset.csv (25,000 × 26)            │
│     → visualizations/*.png (8 gráficos)                         │
└─────────────────────────────────────────────────────────────────┘
```

## 4) Data Issues & Soluciones

| Problema | Solución | Módulo |
|----------|----------|--------|
| `price` en formato "$1,234" | Parseo a float64 | `clean_price()` |
| `amenities` como JSON string | Parseo a lista Python | `clean_amenities()` |
| `host_is_superhost` como "t"/"f" | Conversión a bool | `clean_boolean_columns()` |
| Inconsistencias en categóricas | Normalización + cast | `clean_categorical()` |
| Nulos en bedrooms, beds (~24%) | Imputación por mediana grupal | `impute_missing()` |
| Nulos en review_scores (~20%) | Conservados (info válida) | - |
| Alta dimensionalidad (79 cols) | Selección inteligente | `get_keep_cols()` |

## 5) Hallazgos Principales

### Insight 1: Factores del Precio
| Variable | Correlación | Interpretación |
|----------|-------------|----------------|
| `accommodates` | +0.51 | **Driver principal** |
| `beds` | +0.40 | Infraestructura importa |
| `bedrooms` | +0.36 | Más espacio = más precio |
| `amenities_count` | +0.19 | Efecto moderado |
| `reviews/rating` | ~0 | No afectan precio |

> **Conclusión:** El precio está determinado por **capacidad física**, no por reputación.

### Insight 2: Geografía y Tipo
- **Recoletos** es el barrio más caro (mediana 192€)
- **Puerta del Ángel** es el más económico (mediana 69€)
- Ratio entre extremos: **2.8x**
- Entire home/apt domina el mercado (66.8%)

### Insight 3: Efecto Superhost
| Métrica | Superhost | No-Superhost | Diferencia |
|---------|-----------|--------------|------------|
| Precio (mediana) | 111€ | 104€ | +7% |
| Reseñas (mediana) | 52 | 6 | **+767%** |
| Rating (media) | 4.84 | 4.55 | +6% |
| Amenities | 34 | 24 | +44% |

> **Conclusión:** Superhosts no cobran premium significativo, pero tienen **8x más reservas**. La estrategia es de **volumen**, no de precio.

## 6) Estructura del Proyecto

```
Eda_project/
├── data/
│   ├── raw/
│   │   └── listings.csv              # Dataset original
│   └── processed/
│       └── clean_dataset.csv         # Dataset procesado (26 cols)
├── notebooks/
│   └── eda.ipynb                     # Análisis exploratorio
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuración y columnas
│   ├── io_utils.py                   # Carga/guardado
│   ├── cleaning.py                   # Pipeline de limpieza (9 funciones)
│   ├── features.py                   # Feature engineering (6 features)
│   ├── validation.py                 # Validación post-limpieza
│   ├── analytics.py                  # Tablas resumen y pivots
│   ├── viz.py                        # Visualizaciones (8 gráficos)
│   └── utils.py                      # QC inicial
├── visualizations/                   # Gráficos exportados
│   ├── p1_heatmap_correlacion.png
│   ├── p1_ranking_correlacion_precio.png
│   ├── p2_boxplot_precio_room_type.png
│   ├── p2_barras_precio_barrio.png
│   ├── p2_heatmap_barrio_room_type.png
│   ├── p2_boxplot_price_per_person.png
│   ├── p3_boxplot_superhost.png
│   └── p3_barras_superhost_diff.png
├── main.py                           # Pipeline end-to-end
├── requirements.txt                  # Dependencias
└── README.md
```

### Módulos

| Módulo | Responsabilidad |
|--------|-----------------|
| `config.py` | Rutas, columnas base y opcionales |
| `cleaning.py` | 9 funciones de limpieza orquestadas |
| `features.py` | 6 funciones de feature engineering |
| `validation.py` | Suite de validación post-procesamiento |
| `analytics.py` | 8 funciones de tablas analíticas |
| `viz.py` | 8 funciones de visualización |

## 7) Instalación y Ejecución

### Requisitos
- Python 3.10+
- pip

### Instalación
```bash
# Clonar o descargar el proyecto
cd Eda_project

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Ejecutar pipeline completo
```bash
python main.py
```

### Ejecutar notebook interactivo
```bash
jupyter notebook notebooks/eda.ipynb
```

## 8) Resultados

El pipeline genera:
- **Dataset limpio:** `data/processed/clean_dataset.csv` (25,000 × 26)
- **8 visualizaciones:** en `visualizations/`
- **7 tablas analíticas:** impresas en consola
- **Reporte de validación:** estado PASS/REVIEW

## 9) Conclusiones

### Hallazgo principal
> El precio en Airbnb Madrid está determinado por **factores estructurales** (capacidad, ubicación, tipo) más que por reputación. Ser Superhost atrae demanda pero no justifica un premium en tarifa.

### Recomendaciones para hosts
1. **Maximiza capacidad** si buscas mayores ingresos por noche
2. **Ubicación premium** justifica precios significativamente más altos
3. **Ser Superhost** es estrategia de volumen, no de precio
4. **Amenities** tienen impacto moderado pero positivo

### Próximos pasos
- [ ] Modelado predictivo de precios
- [ ] Análisis temporal (estacionalidad)
- [ ] NLP en descripciones
- [ ] Sistema de recomendación de precios

---

**Proyecto EDA — Máster Data Science & IA, 2025**
