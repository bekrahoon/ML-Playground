import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, 
    r2_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder


def load_dataset(file_path):
    """Загрузка CSV файла"""
    try:
        df = pd.read_csv(file_path)
        return df, None
    except Exception as e:
        return None, str(e)


def get_dataset_info(df):
    """Получение базовой информации о датасете"""
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'describe': df.describe().to_dict(),
    }
    
    # Корреляционная матрица для числовых колонок
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        info['correlation'] = df[numeric_cols].corr().to_dict()
    
    return info


def prepare_data(df, target_column, feature_columns=None):
    """Подготовка данных для обучения"""
    # Если признаки не указаны, используем все колонки кроме целевой
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    # Удаляем строки с пропусками
    df_clean = df[feature_columns + [target_column]].dropna()
    
    X = df_clean[feature_columns]
    y = df_clean[target_column]
    
    # Обработка категориальных признаков
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Обработка категориальной целевой переменной для классификации
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    
    return X, y


def train_model(algorithm, X_train, X_test, y_train, y_test):
    """Обучение модели и получение метрик"""
    models = {
        'linear_regression': LinearRegression(),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'decision_tree_classifier': DecisionTreeClassifier(random_state=42),
        'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
        'random_forest_classifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'random_forest_regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    }
    
    if algorithm not in models:
        raise ValueError(f'Неизвестный алгоритм: {algorithm}')
    
    model = models[algorithm]
    model.fit(X_train, y_train)
    
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Расчет метрик
    metrics = {}
    
    # Определяем тип задачи
    is_classification = algorithm in [
        'logistic_regression', 
        'decision_tree_classifier', 
        'random_forest_classifier'
    ]
    
    if is_classification:
        metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
        metrics['f1_score'] = float(f1_score(y_test, y_pred, average='weighted'))
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    else:
        metrics['mse'] = float(mean_squared_error(y_test, y_pred))
        metrics['rmse'] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['r2_score'] = float(r2_score(y_test, y_pred))
    
    return model, metrics


def save_model(model, file_path):
    """Сохранение модели в pickle файл"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_path):
    """Загрузка модели из pickle файла"""
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(model, X):
    """Получение предсказаний"""
    return model.predict(X)


def train_ml_model(dataset_file, algorithm, target_column, feature_columns=None, test_size=0.2):
    """Полный пайплайн обучения модели"""
    # Загрузка данных
    df, error = load_dataset(dataset_file)
    if error:
        return None, None, f'Ошибка загрузки данных: {error}'
    
    # Подготовка данных
    try:
        X, y = prepare_data(df, target_column, feature_columns)
    except Exception as e:
        return None, None, f'Ошибка подготовки данных: {str(e)}'
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Обучение модели
    try:
        model, metrics = train_model(algorithm, X_train, X_test, y_train, y_test)
    except Exception as e:
        return None, None, f'Ошибка обучения модели: {str(e)}'
    
    return model, metrics, None