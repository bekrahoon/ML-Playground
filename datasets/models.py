import os
from django.db import models
from django.contrib.auth.models import User
from django.core.validators import FileExtensionValidator


def dataset_upload_path(instance, filename):
    return f'datasets/{instance.owner.username}/{filename}'


def model_upload_path(instance, filename):
    return f'models/{instance.owner.username}/{filename}'


class Dataset(models.Model):
    name = models.CharField('Название', max_length=200)
    description = models.TextField('Описание', blank=True)
    file = models.FileField(
        'CSV файл',
        upload_to=dataset_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['csv'])]
    )
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='datasets')
    created_at = models.DateTimeField('Дата создания', auto_now_add=True)
    updated_at = models.DateTimeField('Дата изменения', auto_now=True)
    rows_count = models.IntegerField('Количество строк', null=True, blank=True)
    columns_count = models.IntegerField('Количество столбцов', null=True, blank=True)
    
    class Meta:
        verbose_name = 'Датасет'
        verbose_name_plural = 'Датасеты'
        ordering = ['-created_at']
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)


class MLModel(models.Model):
    ALGORITHM_CHOICES = [
        ('linear_regression', 'Linear Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree_classifier', 'Decision Tree Classifier'),
        ('decision_tree_regressor', 'Decision Tree Regressor'),
        ('random_forest_classifier', 'Random Forest Classifier'),
        ('random_forest_regressor', 'Random Forest Regressor'),
    ]
    
    name = models.CharField('Название модели', max_length=200)
    description = models.TextField('Описание', blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='models')
    algorithm = models.CharField('Алгоритм', max_length=50, choices=ALGORITHM_CHOICES)
    target_column = models.CharField('Целевая переменная', max_length=100)
    feature_columns = models.JSONField('Признаки', default=list)
    
    # Метрики
    accuracy = models.FloatField('Accuracy', null=True, blank=True)
    f1_score = models.FloatField('F1 Score', null=True, blank=True)
    mse = models.FloatField('MSE', null=True, blank=True)
    rmse = models.FloatField('RMSE', null=True, blank=True)
    r2_score = models.FloatField('R2 Score', null=True, blank=True)
    confusion_matrix = models.JSONField('Confusion Matrix', null=True, blank=True)
    
    # Файл модели
    model_file = models.FileField(
        'Файл модели',
        upload_to=model_upload_path,
        validators=[FileExtensionValidator(allowed_extensions=['pkl'])]
    )
    
    # Связь с экспериментом (опционально)
    experiment = models.ForeignKey(
        'Experiment', 
        on_delete=models.CASCADE, 
        related_name='models',
        null=True,
        blank=True
    )
    
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ml_models')
    created_at = models.DateTimeField('Дата создания', auto_now_add=True)
    training_time = models.FloatField('Время обучения (сек)', null=True, blank=True)
    
    class Meta:
        verbose_name = 'ML Модель'
        verbose_name_plural = 'ML Модели'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.name} ({self.get_algorithm_display()})'
    
    def delete(self, *args, **kwargs):
        if self.model_file:
            if os.path.isfile(self.model_file.path):
                os.remove(self.model_file.path)
        super().delete(*args, **kwargs)


class Experiment(models.Model):
    """ML Experiment - сравнение нескольких моделей"""
    
    EXPERIMENT_TYPE_CHOICES = [
        ('manual', 'Ручное сравнение'),
        ('automl', 'AutoML'),
    ]
    
    name = models.CharField('Название эксперимента', max_length=200)
    description = models.TextField('Описание', blank=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='experiments')
    experiment_type = models.CharField('Тип эксперимента', max_length=20, 
                                      choices=EXPERIMENT_TYPE_CHOICES, default='manual')
    
    # Параметры эксперимента
    target_column = models.CharField('Целевая переменная', max_length=100)
    feature_columns = models.JSONField('Признаки', default=list)
    test_size = models.FloatField('Размер тестовой выборки', default=0.2)
    
    # Выбранные алгоритмы для сравнения (для manual режима)
    selected_algorithms = models.JSONField('Выбранные алгоритмы', default=list)
    
    # AutoML параметры
    automl_settings = models.JSONField('Настройки AutoML', null=True, blank=True)
    task_type = models.CharField('Тип задачи', max_length=20, null=True, blank=True)
    
    # Статус
    STATUS_CHOICES = [
        ('pending', 'Ожидание'),
        ('running', 'Обучение'),
        ('completed', 'Завершено'),
        ('failed', 'Ошибка'),
    ]
    status = models.CharField('Статус', max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Результаты
    best_model = models.ForeignKey(
        MLModel,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='best_in_experiments'
    )
    results_summary = models.JSONField('Сводка результатов', null=True, blank=True)
    recommendations = models.JSONField('Рекомендации', null=True, blank=True)
    
    # Метаданные
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name='experiments')
    created_at = models.DateTimeField('Дата создания', auto_now_add=True)
    completed_at = models.DateTimeField('Дата завершения', null=True, blank=True)
    total_training_time = models.FloatField('Общее время обучения (сек)', null=True, blank=True)
    
    class Meta:
        verbose_name = 'Эксперимент'
        verbose_name_plural = 'Эксперименты'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.name} ({self.get_experiment_type_display()} - {self.status})'
    
    def get_models_count(self):
        return self.models.count()
    
    def get_best_algorithm(self):
        if self.best_model:
            return self.best_model.get_algorithm_display()
        return None