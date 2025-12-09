"""
AutoML - Автоматический подбор лучшей ML модели
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
import pickle
import time


def detect_task_type(y):
    """
    Автоматически определяет тип задачи (классификация или регрессия)
    """
    unique_values = len(np.unique(y))
    total_values = len(y)
    
    # Если уникальных значений < 20 и они составляют < 5% от всех данных
    if unique_values < 20 and unique_values / total_values < 0.05:
        return 'classification'
    else:
        return 'regression'


def get_recommended_algorithms(task_type):
    """
    Возвращает рекомендуемые алгоритмы для типа задачи
    """
    if task_type == 'classification':
        return {
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'max_iter': [100, 200, 300]
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }
        }
    else:  # regression
        return {
            'linear_regression': {
                'model': LinearRegression,
                'params': {}
            },
            'decision_tree': {
                'model': DecisionTreeRegressor,
                'params': {
                    'max_depth': [3, 5, 7, 10, None],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            }
        }


def preprocess_data(X, y):
    """
    Автоматическая предобработка данных
    """
    # Копируем данные
    X_processed = X.copy()
    
    # Обработка категориальных признаков
    for col in X_processed.columns:
        if X_processed[col].dtype == 'object':
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    
    # Заполнение пропусков средним значением
    X_processed = X_processed.fillna(X_processed.mean())
    
    # Обработка целевой переменной для классификации
    if y.dtype == 'object':
        le = LabelEncoder()
        y_processed = le.fit_transform(y)
    else:
        y_processed = y
    
    return X_processed, y_processed


def run_automl(dataset_path, target_column, feature_columns=None, test_size=0.2, cv_folds=5):
    """
    Запускает AutoML процесс
    
    Args:
        dataset_path: путь к CSV файлу
        target_column: название целевой колонки
        feature_columns: список признаков (если None - все кроме target)
        test_size: размер тестовой выборки
        cv_folds: количество фолдов для cross-validation
    
    Returns:
        dict: результаты AutoML
    """
    results = {
        'success': False,
        'task_type': None,
        'models': [],
        'best_model': None,
        'preprocessing_info': {},
        'recommendations': []
    }
    
    try:
        # Загрузка данных
        df = pd.read_csv(dataset_path)
        
        # Проверка наличия целевой переменной
        if target_column not in df.columns:
            results['error'] = f"Колонка '{target_column}' не найдена"
            return results
        
        # Разделение на признаки и целевую переменную
        if feature_columns:
            X = df[feature_columns]
        else:
            X = df.drop(columns=[target_column])
        
        y = df[target_column]
        
        # Информация о предобработке
        results['preprocessing_info'] = {
            'original_features': len(X.columns),
            'samples': len(X),
            'missing_values_before': X.isnull().sum().sum(),
            'categorical_features': len(X.select_dtypes(include=['object']).columns)
        }
        
        # Предобработка
        X_processed, y_processed = preprocess_data(X, y)
        
        results['preprocessing_info']['missing_values_after'] = 0  # После fillna
        
        # Определение типа задачи
        task_type = detect_task_type(y_processed)
        results['task_type'] = task_type
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, random_state=42
        )
        
        # Масштабирование признаков (для некоторых алгоритмов)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Получаем рекомендуемые алгоритмы
        algorithms = get_recommended_algorithms(task_type)
        
        # Обучаем каждую модель с GridSearch
        for algo_name, algo_config in algorithms.items():
            print(f"AutoML: Обучение {algo_name}...")
            
            start_time = time.time()
            
            model_class = algo_config['model']
            param_grid = algo_config['params']
            
            # Используем scaled данные для логистической регрессии
            if algo_name == 'logistic_regression':
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # GridSearch для подбора гиперпараметров
            if param_grid:
                grid_search = GridSearchCV(
                    model_class(),
                    param_grid,
                    cv=min(cv_folds, len(X_train) // 2),  # Не больше половины данных
                    scoring='accuracy' if task_type == 'classification' else 'r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train_use, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            else:
                # Для моделей без гиперпараметров
                best_model = model_class()
                best_model.fit(X_train_use, y_train)
                best_params = {}
            
            # Предсказания
            y_pred = best_model.predict(X_test_use)
            
            # Вычисление метрик
            training_time = time.time() - start_time
            
            if task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train_use, y_train,
                    cv=min(cv_folds, len(X_train) // 2),
                    scoring='accuracy'
                )
                
                model_result = {
                    'algorithm': algo_name,
                    'algorithm_display': algo_name.replace('_', ' ').title(),
                    'model': best_model,
                    'best_params': best_params,
                    'training_time': training_time,
                    'accuracy': float(accuracy),
                    'f1_score': float(f1),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'score': float(accuracy)  # Для сравнения
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    best_model, X_train_use, y_train,
                    cv=min(cv_folds, len(X_train) // 2),
                    scoring='r2'
                )
                
                model_result = {
                    'algorithm': algo_name,
                    'algorithm_display': algo_name.replace('_', ' ').title(),
                    'model': best_model,
                    'best_params': best_params,
                    'training_time': training_time,
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'r2_score': float(r2),
                    'cv_mean': float(cv_scores.mean()),
                    'cv_std': float(cv_scores.std()),
                    'score': float(r2)  # Для сравнения
                }
            
            results['models'].append(model_result)
        
        # Определяем лучшую модель
        best_model_result = max(results['models'], key=lambda x: x['score'])
        results['best_model'] = best_model_result
        
        # Генерируем рекомендации
        results['recommendations'] = generate_recommendations(
            results['models'],
            best_model_result,
            task_type,
            results['preprocessing_info']
        )
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        print(f"AutoML Error: {e}")
    
    return results


def generate_recommendations(models, best_model, task_type, preprocessing_info):
    """
    Генерирует рекомендации на основе результатов AutoML
    """
    recommendations = []
    
    # Рекомендация по лучшей модели
    recommendations.append({
        'type': 'best_model',
        'text': f"Лучшая модель: {best_model['algorithm_display']} "
                f"({'Accuracy' if task_type == 'classification' else 'R²'}: {best_model['score']:.4f})"
    })
    
    # Рекомендации по гиперпараметрам
    if best_model['best_params']:
        params_text = ', '.join([f"{k}={v}" for k, v in best_model['best_params'].items()])
        recommendations.append({
            'type': 'hyperparameters',
            'text': f"Оптимальные гиперпараметры: {params_text}"
        })
    
    # Рекомендации по данным
    if preprocessing_info.get('missing_values_before', 0) > 0:
        recommendations.append({
            'type': 'data_quality',
            'text': f"В данных было {preprocessing_info['missing_values_before']} пропущенных значений. "
                    "Рекомендуется улучшить качество данных."
        })
    
    if preprocessing_info.get('samples', 0) < 100:
        recommendations.append({
            'type': 'data_size',
            'text': "Размер датасета небольшой. Для улучшения качества модели рекомендуется собрать больше данных."
        })
    
    # Сравнение моделей
    if len(models) > 1:
        sorted_models = sorted(models, key=lambda x: x['score'], reverse=True)
        second_best = sorted_models[1]
        score_diff = (sorted_models[0]['score'] - sorted_models[1]['score']) * 100
        
        if score_diff < 5:
            recommendations.append({
                'type': 'alternatives',
                'text': f"Модель {second_best['algorithm_display']} показала близкий результат "
                        f"(разница {score_diff:.1f}%), но обучается быстрее."
            })
    
    return recommendations