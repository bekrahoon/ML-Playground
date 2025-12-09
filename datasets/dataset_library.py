"""
Dataset Library - Встроенные датасеты для быстрого старта
"""
import pandas as pd
import os
from django.conf import settings
from sklearn import datasets as sklearn_datasets


# Каталог встроенных датасетов
BUILT_IN_DATASETS = {
    'iris': {
        'name': 'Iris Flower Dataset',
        'description': 'Классический датасет для классификации видов ирисов по характеристикам цветков. '
                      'Содержит 150 образцов 3 видов ирисов (Setosa, Versicolor, Virginica).',
        'rows': 150,
        'columns': 5,
        'task_type': 'classification',
        'target': 'species',
        'features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'recommended_algorithms': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'difficulty': 'beginner',
        'tags': ['classification', 'multiclass', 'small', 'beginner'],
        'source': 'sklearn',
        'icon': 'flower',
        'use_cases': ['Изучение основ ML', 'Классификация', 'Визуализация данных']
    },
    'titanic': {
        'name': 'Titanic Survival Dataset',
        'description': 'Датасет о пассажирах Титаника. Задача - предсказать выживет ли пассажир '
                      'на основе возраста, пола, класса каюты и других признаков.',
        'rows': 891,
        'columns': 12,
        'task_type': 'classification',
        'target': 'Survived',
        'features': ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
        'recommended_algorithms': ['Logistic Regression', 'Random Forest', 'Decision Tree'],
        'difficulty': 'beginner',
        'tags': ['classification', 'binary', 'missing_values', 'categorical'],
        'source': 'csv',
        'icon': 'ship',
        'use_cases': ['Бинарная классификация', 'Работа с категориальными данными', 'Обработка пропусков']
    },
    'wine': {
        'name': 'Wine Quality Dataset',
        'description': 'Датасет о качестве вин. Содержит химические характеристики вин и их качество. '
                      'Можно использовать для классификации или регрессии.',
        'rows': 178,
        'columns': 14,
        'task_type': 'classification',
        'target': 'target',
        'features': ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 
                    'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
                    'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline'],
        'recommended_algorithms': ['Random Forest', 'Logistic Regression', 'Decision Tree'],
        'difficulty': 'beginner',
        'tags': ['classification', 'multiclass', 'wine', 'chemistry'],
        'source': 'sklearn',
        'icon': 'wine',
        'use_cases': ['Классификация качества', 'Feature engineering', 'Мультикласс классификация']
    },
    'diabetes': {
        'name': 'Diabetes Dataset',
        'description': 'Датасет для предсказания прогрессирования диабета через год после baseline. '
                      'Содержит 10 базовых переменных: возраст, пол, индекс массы тела и т.д.',
        'rows': 442,
        'columns': 11,
        'task_type': 'regression',
        'target': 'target',
        'features': ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'],
        'recommended_algorithms': ['Linear Regression', 'Random Forest Regressor', 'Decision Tree Regressor'],
        'difficulty': 'intermediate',
        'tags': ['regression', 'healthcare', 'continuous'],
        'source': 'sklearn',
        'icon': 'heart-pulse',
        'use_cases': ['Регрессия', 'Предсказание медицинских показателей', 'Feature selection']
    },
    'california_housing': {
        'name': 'California Housing Dataset',
        'description': 'Датасет о ценах на жилье в Калифорнии. Содержит медианную стоимость домов '
                      'и характеристики районов (население, доход, локация и т.д.).',
        'rows': 20640,
        'columns': 9,
        'task_type': 'regression',
        'target': 'MedHouseVal',
        'features': ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                    'AveOccup', 'Latitude', 'Longitude'],
        'recommended_algorithms': ['Random Forest Regressor', 'Linear Regression', 'Decision Tree Regressor'],
        'difficulty': 'intermediate',
        'tags': ['regression', 'real_estate', 'large', 'geo'],
        'source': 'sklearn',
        'icon': 'house',
        'use_cases': ['Предсказание цен', 'Регрессия', 'Работа с географическими данными']
    },
    'breast_cancer': {
        'name': 'Breast Cancer Wisconsin Dataset',
        'description': 'Медицинский датасет для диагностики рака груди. Содержит характеристики '
                      'клеток опухоли. Задача - классифицировать опухоль как злокачественную или доброкачественную.',
        'rows': 569,
        'columns': 31,
        'task_type': 'classification',
        'target': 'target',
        'features': ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'],
        'recommended_algorithms': ['Logistic Regression', 'Random Forest', 'Decision Tree'],
        'difficulty': 'intermediate',
        'tags': ['classification', 'binary', 'healthcare', 'medical'],
        'source': 'sklearn',
        'icon': 'hospital',
        'use_cases': ['Медицинская диагностика', 'Бинарная классификация', 'Feature importance']
    }
}


def get_dataset_info(dataset_id):
    """Получить информацию о датасете"""
    return BUILT_IN_DATASETS.get(dataset_id)


def get_all_datasets():
    """Получить все доступные датасеты"""
    return BUILT_IN_DATASETS


def load_built_in_dataset(dataset_id):
    """
    Загрузить встроенный датасет и вернуть как pandas DataFrame
    
    Args:
        dataset_id: идентификатор датасета
    
    Returns:
        pd.DataFrame: загруженный датасет
    """
    if dataset_id not in BUILT_IN_DATASETS:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    info = BUILT_IN_DATASETS[dataset_id]
    
    if info['source'] == 'sklearn':
        # Загрузка из sklearn
        if dataset_id == 'iris':
            data = sklearn_datasets.load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['species'] = pd.Categorical.from_codes(data.target, data.target_names)
            
        elif dataset_id == 'wine':
            data = sklearn_datasets.load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif dataset_id == 'diabetes':
            data = sklearn_datasets.load_diabetes()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
        elif dataset_id == 'california_housing':
            data = sklearn_datasets.fetch_california_housing()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['MedHouseVal'] = data.target
            
        elif dataset_id == 'breast_cancer':
            data = sklearn_datasets.load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
    elif info['source'] == 'csv':
        # Загрузка из CSV (для Titanic)
        if dataset_id == 'titanic':
            df = create_titanic_dataset()
    
    return df


def create_titanic_dataset():
    """Создает базовый датасет Titanic"""
    # Упрощенная версия датасета Titanic
    data = {
        'PassengerId': range(1, 892),
        'Survived': [0, 1, 1, 1, 0] * 178 + [0],
        'Pclass': [3, 1, 3, 1, 3] * 178 + [3],
        'Name': [f'Passenger {i}' for i in range(1, 892)],
        'Sex': ['male', 'female'] * 445 + ['male'],
        'Age': [22, 38, 26, 35, 35] * 178 + [22],
        'SibSp': [1, 1, 0, 1, 0] * 178 + [1],
        'Parch': [0, 0, 0, 0, 0] * 178 + [0],
        'Ticket': [f'A{i}' for i in range(1, 892)],
        'Fare': [7.25, 71.28, 7.92, 53.1, 8.05] * 178 + [7.25],
        'Cabin': [''] * 891,
        'Embarked': ['S', 'C', 'S', 'S', 'S'] * 178 + ['S']
    }
    
    df = pd.DataFrame(data)
    return df


def import_dataset_for_user(dataset_id, user):
    """
    Импортирует встроенный датасет для пользователя
    
    Args:
        dataset_id: идентификатор датасета
        user: пользователь Django
    
    Returns:
        Dataset: созданный объект датасета
    """
    from .models import Dataset
    from django.core.files.base import ContentFile
    import tempfile
    
    # Получаем информацию о датасете
    info = get_dataset_info(dataset_id)
    if not info:
        raise ValueError(f"Dataset {dataset_id} not found")
    
    # Загружаем данные
    df = load_built_in_dataset(dataset_id)
    
    # Создаем временный CSV файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        
        with open(tmp.name, 'rb') as f:
            csv_content = f.read()
    
    # Создаем объект Dataset
    dataset = Dataset(
        name=info['name'],
        description=f"{info['description']}\n\n"
                   f"Тип задачи: {info['task_type']}\n"
                   f"Целевая переменная: {info['target']}\n"
                   f"Рекомендуемые алгоритмы: {', '.join(info['recommended_algorithms'])}",
        owner=user,
        rows_count=info['rows'],
        columns_count=info['columns']
    )
    
    # Сохраняем файл
    dataset.file.save(
        f'{dataset_id}_dataset.csv',
        ContentFile(csv_content),
        save=False
    )
    
    dataset.save()
    
    # Удаляем временный файл
    os.unlink(tmp.name)
    
    return dataset