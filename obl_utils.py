"""Código común del Obligatorio MLSI 2026 S2 (predicción de `opinion`).

Contiene la carga de datos, el preprocesamiento y la evaluación con k-fold
cross-validation. Todas las notebooks de modelos lo importan, así el
preprocesamiento y la forma de evaluar son idénticos para todos los modelos.

Uso típico en una notebook:

    from obl_utils import *
    train, test = cargar_datos()
    X, y, X_test, ids_test = preparar_datos(train, test)
    resultado = evaluar(DecisionTreeClassifier(random_state=SEED), X, y)
"""

import os

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SEED = 42                # semilla única para todo el obligatorio (folds, encoders, modelos)
TARGET = 'opinion'
N_FOLDS = 5
UMBRAL_MONTO = 1000      # budget/revenue por debajo de esto se consideran faltantes
URL_DATOS = 'https://raw.githubusercontent.com/juanmareolon/OBL_ML/main/data/'

METRICAS = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'auc': 'roc_auc',
}


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------

def cargar_datos():
    """Lee train y test desde data/ si existe; si no (ej. en Colab), desde GitHub."""
    if os.path.exists('data/movies_train.csv'):
        base = 'data/'
    else:
        print('No está la carpeta data, se leen los CSV desde GitHub')
        base = URL_DATOS
    train = pd.read_csv(base + 'movies_train.csv')
    test = pd.read_csv(base + 'movies_test.csv')
    return train, test


# ---------------------------------------------------------------------------
# 2. Preprocesamiento
# ---------------------------------------------------------------------------

def limpiar(df):
    """Capa 1: limpieza fija. No aprende nada de los datos, se aplica igual a train y test."""
    out = pd.DataFrame(index=df.index)

    # budget / revenue: 0 y valores absurdos (< 1000) -> NaN, con indicador de faltante
    for col in ['budget', 'revenue']:
        valor = df[col].where(df[col] >= UMBRAL_MONTO)
        out[f'{col}_desconocido'] = valor.isna().astype(int)
        out[f'log_{col}'] = np.log1p(valor)

    # ROI: solo si budget y revenue son conocidos
    budget = df['budget'].where(df['budget'] >= UMBRAL_MONTO)
    revenue = df['revenue'].where(df['revenue'] >= UMBRAL_MONTO)
    out['log_roi'] = np.log1p(revenue / budget)

    # runtime: 0 -> NaN
    out['runtime'] = df['runtime'].where(df['runtime'] > 0)

    # vote_count (los 0 son reales) y popularity: log1p
    out['log_vote_count'] = np.log1p(df['vote_count'])
    out['log_popularity'] = np.log1p(df['popularity'])

    # Fecha de estreno -> año y mes
    fecha = pd.to_datetime(df['release_date'], errors='coerce')
    out['anio'] = fecha.dt.year
    out['mes'] = fecha.dt.month

    # keywords: indicador de faltante y cantidad de palabras
    out['sin_keywords'] = df['keywords'].isna().astype(int)
    out['cant_keywords'] = df['keywords'].fillna('').str.split().str.len()

    # Categóricas (se codifican dentro del pipeline)
    out['main_genre_top10'] = df['main_genre_top10']
    out['original_language'] = df['original_language']
    out['director'] = df['director'].fillna('desconocido')
    return out


def preparar_datos(train, test):
    """Aplica `limpiar` y separa features, target e ids del test."""
    X = limpiar(train)
    y = train[TARGET]
    X_test = limpiar(test)
    ids_test = test['id']
    return X, y, X_test, ids_test


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Reemplaza cada categoría por su cantidad de apariciones en el train (0 si no aparece)."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columnas_ = list(X.columns)
        self.conteos_ = {c: X[c].value_counts() for c in self.columnas_}
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columnas_)
        return np.column_stack([
            X[c].map(self.conteos_[c]).fillna(0).to_numpy(dtype=float) for c in self.columnas_
        ])

    def get_feature_names_out(self, input_features=None):
        return np.array([f'{c}_freq' for c in self.columnas_])


def crear_target_encoder():
    """TargetEncoder reproducible. Desde sklearn 1.9 la semilla se pasa vía `cv`;
    en versiones anteriores (Colab trae 1.6) vía `random_state`."""
    version = tuple(int(v) for v in sklearn.__version__.split('.')[:2])
    if version >= (1, 9):
        return TargetEncoder(target_type='binary', cv=KFold(5, shuffle=True, random_state=SEED))
    return TargetEncoder(target_type='binary', random_state=SEED)


COLS_NUMERICAS = [
    'log_budget', 'log_revenue', 'log_roi', 'runtime', 'log_vote_count', 'log_popularity',
    'anio', 'mes', 'cant_keywords',
]
COLS_BINARIAS = ['budget_desconocido', 'revenue_desconocido', 'sin_keywords']
VARIANTES_DIRECTOR = ('ninguno', 'freq', 'freq+target')


def crear_preprocesador(director='freq+target'):
    """Capa 2: ColumnTransformer que se ajusta dentro de cada fold.

    director: 'ninguno' | 'freq' | 'freq+target'
    """
    if director not in VARIANTES_DIRECTOR:
        raise ValueError(f'director inválido: {director}. Opciones: {VARIANTES_DIRECTOR}')

    numericas = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler()),
    ])
    transformadores = [
        ('num', numericas, COLS_NUMERICAS),
        ('bin', 'passthrough', COLS_BINARIAS),
        ('genero', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['main_genre_top10']),
        ('idioma', OneHotEncoder(min_frequency=20, handle_unknown='infrequent_if_exist',
                                 sparse_output=False), ['original_language']),
    ]
    if director in ('freq', 'freq+target'):
        transformadores.append(('director_freq', Pipeline([
            ('freq', FrequencyEncoder()),
            ('escalar', StandardScaler()),
        ]), ['director']))
    if director == 'freq+target':
        transformadores.append(('director_target', crear_target_encoder(), ['director']))

    return ColumnTransformer(transformadores, remainder='drop', verbose_feature_names_out=False)


def crear_pipeline(modelo, director='freq+target'):
    """Une el preprocesador y el modelo, así la CV reajusta todo en cada fold."""
    return Pipeline([
        ('prep', crear_preprocesador(director)),
        ('modelo', modelo),
    ])


# ---------------------------------------------------------------------------
# 3. Validación (k-fold estratificado) y métricas
# ---------------------------------------------------------------------------

def crear_cv():
    """K-fold estratificado: los mismos 5 folds para todos los modelos."""
    return StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def evaluar(modelo, X, y, director='freq+target', **info):
    """Evalúa un modelo con k-fold CV y devuelve un dict con las métricas.

    Para cada métrica devuelve el promedio y el desvío entre folds en validación,
    y el AUC en train para detectar sobreajuste (train >> validación).

    `info` permite agregar columnas extra al resultado (ej. los hiperparámetros
    probados), así cada llamada es una fila de la tabla de resultados:

        filas = []
        for d in [3, 5, 8]:
            filas.append(evaluar(DecisionTreeClassifier(max_depth=d), X, y, max_depth=d))
        tabla = pd.DataFrame(filas)
    """
    pipe = crear_pipeline(modelo, director)
    cv = cross_validate(pipe, X, y, cv=crear_cv(), scoring=METRICAS,
                        return_train_score=True, n_jobs=-1)

    fila = {'modelo': type(modelo).__name__, 'director': director, **info}
    for nombre in METRICAS:
        fila[nombre] = float(cv[f'test_{nombre}'].mean())
        fila[f'{nombre}_std'] = float(cv[f'test_{nombre}'].std())
    fila['auc_train'] = float(cv['train_auc'].mean())
    return fila


def matriz_confusion(modelo, X, y, director='freq+target', titulo=None, ax=None):
    """Matriz de confusión con las predicciones de validación de cada fold
    (cada película se predice con el modelo que no la vio al entrenar)."""
    pipe = crear_pipeline(modelo, director)
    y_pred = cross_val_predict(pipe, X, y, cv=crear_cv(), n_jobs=-1)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['0 (negativa)', '1 (positiva)'])
    disp.plot(cmap='Blues', values_format='d', ax=ax, colorbar=False)
    if titulo:
        disp.ax_.set_title(titulo)
    return cm


def generar_submission(modelo, X, y, X_test, ids_test, archivo, director='freq+target'):
    """Entrena con todo el train y guarda las PROBABILIDADES de la clase 1 para Kaggle."""
    pipe = crear_pipeline(modelo, director)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X_test)[:, 1]
    sub = pd.DataFrame({'id': ids_test, TARGET: proba})
    sub.to_csv(archivo, index=False)
    print(f'Guardado {archivo} ({len(sub)} filas)')
    return sub
