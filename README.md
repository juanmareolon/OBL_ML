# Obligatorio Machine Learning para Sistemas Inteligentes (2026, S2)

Obligatorio de la materia Machine Learning para Sistemas Inteligentes, sobre un conjunto de datos de películas. Tiene dos tareas:

1. **Clasificación binaria de `opinion`**: predecir si una película recibe opinión positiva. Se evalúa con AUC en una competencia privada de Kaggle.
2. **Clasificación multiclase del género (NLP)**: predecir `main_genre_top10` a partir del título, la sinopsis y las palabras clave.

## Modelo final

**Stacking** de regresión logística, Random Forest, Gradient Boosting y red neuronal, con una regresión logística como meta-modelo.
AUC de validación cruzada: **0,840 ± 0,010**. AUC en Kaggle (público): **0,846**.

## Notebooks

Todas usan la misma validación cruzada estratificada de 5 folds (semilla 42) y el preprocesamiento común de `obl_utils.py`, dentro de un pipeline.

| Notebook | Contenido | AUC validación | AUC Kaggle |
|---|---|---|---|
| `Arbol_Clasificacion.ipynb` | Árbol de decisión, grilla de 108 combinaciones | 0,769 ± 0,013 | 0,771 |
| `Regresion_Logistica.ipynb` | Regresión logística (C y L1/L2), coeficientes | 0,815 ± 0,009 | 0,819 |
| `Random_Forest.ipynb` | Random Forest, grilla de 48 combinaciones | 0,825 ± 0,011 | 0,844 |
| `Boosting.ipynb` | AdaBoost y Gradient Boosting | Ada 0,831 ± 0,013 / GB 0,836 ± 0,013 | Ada 0,834 / GB 0,841 |
| `Red_Neuronal.ipynb` | Perceptrón multicapa en Keras con early stopping | 0,832 ± 0,011 | 0,837 |
| `Stacking.ipynb` | Componente de investigación: stacking y comparación formal (test t corregido de Nadeau y Bengio) | **0,840 ± 0,010** | **0,846** |
| `Feature_Engineering.ipynb` | Ablación: impacto de cada grupo de variables en la logística y en Gradient Boosting | — | — |
| `NLP_Genero.ipynb` | Tarea 2: preprocesamiento de texto, TF-IDF y comparación de clasificadores | F1 macro 0,336 ± 0,034 | — |

