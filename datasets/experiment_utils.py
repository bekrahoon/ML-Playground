"""
Утилиты для ML Experiments - сравнение моделей
"""
import time
import pandas as pd
from .ml_utils import train_ml_model


def run_experiment(experiment, dataset, feature_columns):
    """
    Запускает эксперимент - обучает несколько моделей и сравнивает их
    
    Args:
        experiment: объект Experiment
        dataset: объект Dataset
        feature_columns: список выбранных признаков
    
    Returns:
        dict: результаты эксперимента
    """
    from .models import MLModel
    import pickle
    import tempfile
    import os
    from django.core.files.base import ContentFile
    from django.utils import timezone
    
    experiment.status = 'running'
    experiment.save()
    
    results = []
    total_time = 0
    trained_models = []
    
    try:
        # Обучаем каждый алгоритм
        for algorithm in experiment.selected_algorithms:
            print(f"Обучение {algorithm}...")
            
            start_time = time.time()
            
            # Обучаем модель
            model_obj, metrics, error = train_ml_model(
                dataset.file.path,
                algorithm,
                experiment.target_column,
                feature_columns,
                test_size=experiment.test_size
            )
            
            training_time = time.time() - start_time
            total_time += training_time
            
            if error:
                print(f"Ошибка при обучении {algorithm}: {error}")
                continue
            
            # Создаем объект MLModel
            ml_model = MLModel(
                name=f"{experiment.name} - {dict(MLModel.ALGORITHM_CHOICES)[algorithm]}",
                description=f"Модель из эксперимента {experiment.name}",
                dataset=dataset,
                algorithm=algorithm,
                target_column=experiment.target_column,
                feature_columns=feature_columns,
                owner=experiment.owner,
                experiment=experiment,
                training_time=training_time
            )
            
            # Сохраняем метрики
            if 'accuracy' in metrics:
                ml_model.accuracy = metrics['accuracy']
            if 'f1_score' in metrics:
                ml_model.f1_score = metrics['f1_score']
            if 'mse' in metrics:
                ml_model.mse = metrics['mse']
            if 'rmse' in metrics:
                ml_model.rmse = metrics['rmse']
            if 'r2_score' in metrics:
                ml_model.r2_score = metrics['r2_score']
            if 'confusion_matrix' in metrics:
                ml_model.confusion_matrix = metrics['confusion_matrix']
            
            # Сохраняем pickle файл модели
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                pickle.dump(model_obj, tmp)
                tmp.flush()
                
                with open(tmp.name, 'rb') as f:
                    ml_model.model_file.save(
                        f'{ml_model.name}.pkl',
                        ContentFile(f.read()),
                        save=False
                    )
                
                os.unlink(tmp.name)
            
            ml_model.save()
            trained_models.append(ml_model)
            
            # Собираем результаты
            result = {
                'algorithm': algorithm,
                'algorithm_display': dict(MLModel.ALGORITHM_CHOICES)[algorithm],
                'training_time': training_time,
                'metrics': metrics,
                'model_id': ml_model.id
            }
            results.append(result)
        
        # Определяем лучшую модель
        best_model = find_best_model(trained_models)
        
        # Сохраняем результаты
        experiment.status = 'completed'
        experiment.completed_at = timezone.now()
        experiment.total_training_time = total_time
        experiment.best_model = best_model
        experiment.results_summary = {
            'models_trained': len(results),
            'total_time': total_time,
            'results': results
        }
        experiment.save()
        
        return {
            'success': True,
            'results': results,
            'best_model': best_model,
            'total_time': total_time
        }
        
    except Exception as e:
        experiment.status = 'failed'
        experiment.results_summary = {'error': str(e)}
        experiment.save()
        
        return {
            'success': False,
            'error': str(e)
        }


def find_best_model(models):
    """
    Определяет лучшую модель из списка на основе метрик
    
    Args:
        models: список объектов MLModel
    
    Returns:
        MLModel: лучшая модель
    """
    if not models:
        return None
    
    # Определяем тип задачи (классификация или регрессия)
    is_classification = any(m.accuracy is not None for m in models)
    
    if is_classification:
        # Для классификации выбираем по accuracy
        best = max(
            [m for m in models if m.accuracy is not None],
            key=lambda m: m.accuracy,
            default=None
        )
    else:
        # Для регрессии выбираем по R2 (или минимальному RMSE)
        models_with_r2 = [m for m in models if m.r2_score is not None]
        if models_with_r2:
            best = max(models_with_r2, key=lambda m: m.r2_score)
        else:
            # Если нет R2, выбираем по минимальному RMSE
            models_with_rmse = [m for m in models if m.rmse is not None]
            if models_with_rmse:
                best = min(models_with_rmse, key=lambda m: m.rmse)
            else:
                best = models[0]
    
    return best


def get_comparison_data(experiment):
    """
    Получает данные для сравнения моделей в эксперименте
    
    Args:
        experiment: объект Experiment
    
    Returns:
        dict: данные для визуализации
    """
    models = experiment.models.all()
    
    if not models:
        return None
    
    # Определяем тип задачи
    is_classification = any(m.accuracy is not None for m in models)
    
    comparison = {
        'labels': [m.get_algorithm_display() for m in models],
        'training_times': [m.training_time for m in models],
        'is_classification': is_classification
    }
    
    if is_classification:
        comparison['accuracy'] = [m.accuracy or 0 for m in models]
        comparison['f1_score'] = [m.f1_score or 0 for m in models]
    else:
        comparison['r2_score'] = [m.r2_score or 0 for m in models]
        comparison['rmse'] = [m.rmse or 0 for m in models]
        comparison['mse'] = [m.mse or 0 for m in models]
    
    return comparison