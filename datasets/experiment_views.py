"""
Views для ML Experiments
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden, JsonResponse
from .models import Dataset, Experiment, MLModel
from .forms import ExperimentForm
from .experiment_utils import run_experiment, get_comparison_data
from .automl_utils import run_automl
from .dataset_library import get_all_datasets, get_dataset_info, import_dataset_for_user
import pickle
import tempfile
import os
from django.core.files.base import ContentFile
from django.utils import timezone


def convert_numpy_types(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON"""
    import numpy as np
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@login_required
def experiment_list(request):
    """Список всех экспериментов пользователя"""
    experiments = Experiment.objects.filter(owner=request.user)
    return render(request, 'datasets/experiment_list.html', {
        'experiments': experiments
    })


@login_required
def experiment_create(request, dataset_pk):
    """Создание нового эксперимента"""
    dataset = get_object_or_404(Dataset, pk=dataset_pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    if request.method == 'POST':
        form = ExperimentForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            # Получаем выбранные признаки
            feature_columns = request.POST.getlist('feature_columns')
            algorithms = form.cleaned_data['algorithms']
            
            if not feature_columns:
                messages.error(request, 'Выберите хотя бы один признак!')
                return render(request, 'datasets/experiment_form.html', {
                    'form': form,
                    'dataset': dataset,
                    'columns': get_columns(dataset)
                })
            
            if not algorithms:
                messages.error(request, 'Выберите хотя бы один алгоритм!')
                return render(request, 'datasets/experiment_form.html', {
                    'form': form,
                    'dataset': dataset,
                    'columns': get_columns(dataset)
                })
            
            # Создаем эксперимент
            experiment = form.save(commit=False)
            experiment.dataset = dataset
            experiment.owner = request.user
            experiment.feature_columns = feature_columns
            experiment.selected_algorithms = algorithms
            experiment.status = 'pending'
            experiment.save()
            
            # Запускаем обучение
            messages.info(request, 'Эксперимент запущен! Обучаем модели...')
            
            result = run_experiment(experiment, dataset, feature_columns)
            
            if result['success']:
                messages.success(request, 
                    f'Эксперимент завершен! Обучено {len(result["results"])} моделей за {result["total_time"]:.2f} сек.')
                return redirect('datasets:experiment_detail', pk=experiment.pk)
            else:
                messages.error(request, f'Ошибка: {result["error"]}')
                return redirect('datasets:experiment_detail', pk=experiment.pk)
    else:
        form = ExperimentForm(dataset=dataset)
    
    return render(request, 'datasets/experiment_form.html', {
        'form': form,
        'dataset': dataset,
        'columns': get_columns(dataset)
    })


@login_required
def experiment_detail(request, pk):
    """Детальная информация об эксперименте"""
    import json
    
    experiment = get_object_or_404(Experiment, pk=pk)
    
    # Проверка доступа
    if experiment.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому эксперименту')
    
    # Получаем данные для сравнения
    comparison_data = get_comparison_data(experiment)
    
    # Получаем все модели эксперимента
    models = experiment.models.all().order_by('-accuracy', '-r2_score')
    
    return render(request, 'datasets/experiment_detail.html', {
        'experiment': experiment,
        'models': models,
        'comparison_data': comparison_data,
        'comparison_data_json': json.dumps(comparison_data) if comparison_data else None
    })


@login_required
def experiment_delete(request, pk):
    """Удаление эксперимента"""
    experiment = get_object_or_404(Experiment, pk=pk)
    
    # Проверка доступа
    if experiment.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому эксперименту')
    
    if request.method == 'POST':
        dataset_pk = experiment.dataset.pk
        experiment.delete()
        messages.success(request, 'Эксперимент успешно удален!')
        return redirect('datasets:dataset_detail', pk=dataset_pk)
    
    return render(request, 'datasets/experiment_confirm_delete.html', {
        'experiment': experiment
    })


@login_required
def experiment_compare_json(request, pk):
    """API endpoint для получения данных сравнения в JSON"""
    experiment = get_object_or_404(Experiment, pk=pk)
    
    # Проверка доступа
    if experiment.owner != request.user:
        return JsonResponse({'error': 'Access denied'}, status=403)
    
    comparison_data = get_comparison_data(experiment)
    
    return JsonResponse(comparison_data, safe=False)


def get_columns(dataset):
    """Получение списка колонок датасета"""
    try:
        import pandas as pd
        df = pd.read_csv(dataset.file.path)
        return list(df.columns)
    except Exception:
        return []


@login_required
def automl_create(request, dataset_pk):
    """Создание AutoML эксперимента"""
    dataset = get_object_or_404(Dataset, pk=dataset_pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description', '')
        target_column = request.POST.get('target_column')
        feature_columns = request.POST.getlist('feature_columns')
        test_size = float(request.POST.get('test_size', 0.2))
        cv_folds = int(request.POST.get('cv_folds', 5))
        
        if not name or not target_column:
            messages.error(request, 'Заполните обязательные поля!')
            return render(request, 'datasets/automl_form.html', {
                'dataset': dataset,
                'columns': get_columns(dataset)
            })
        
        # Создаем эксперимент
        experiment = Experiment(
            name=name,
            description=description,
            dataset=dataset,
            target_column=target_column,
            feature_columns=feature_columns if feature_columns else None,
            test_size=test_size,
            owner=request.user,
            experiment_type='automl',
            status='pending',
            automl_settings={
                'cv_folds': cv_folds
            }
        )
        experiment.save()
        
        # Запускаем AutoML
        messages.info(request, '🤖 AutoML запущен! Подбираем лучшую модель...')
        
        automl_results = run_automl(
            dataset.file.path,
            target_column,
            feature_columns if feature_columns else None,
            test_size,
            cv_folds
        )
        
        if automl_results['success']:
            # Сохраняем результаты (конвертируем numpy типы)
            experiment.status = 'completed'
            experiment.completed_at = timezone.now()
            experiment.task_type = automl_results['task_type']
            experiment.recommendations = convert_numpy_types(automl_results['recommendations'])
            experiment.results_summary = convert_numpy_types({
                'preprocessing': automl_results['preprocessing_info'],
                'models_count': len(automl_results['models'])
            })
            
            total_time = sum(float(m['training_time']) for m in automl_results['models'])
            experiment.total_training_time = float(total_time)
            
            # Сохраняем модели
            for model_result in automl_results['models']:
                ml_model = MLModel(
                    name=f"{experiment.name} - {model_result['algorithm_display']}",
                    description=f"AutoML модель. Гиперпараметры: {model_result['best_params']}",
                    dataset=dataset,
                    algorithm=model_result['algorithm'],
                    target_column=target_column,
                    feature_columns=feature_columns if feature_columns else list(model_result['model'].feature_names_in_ if hasattr(model_result['model'], 'feature_names_in_') else []),
                    owner=request.user,
                    experiment=experiment,
                    training_time=model_result['training_time']
                )
                
                # Метрики (конвертируем в float)
                if 'accuracy' in model_result:
                    ml_model.accuracy = float(model_result['accuracy'])
                    ml_model.f1_score = float(model_result['f1_score'])
                if 'r2_score' in model_result:
                    ml_model.r2_score = float(model_result['r2_score'])
                    ml_model.mse = float(model_result['mse'])
                    ml_model.rmse = float(model_result['rmse'])
                
                # Сохраняем pickle файл
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
                    pickle.dump(model_result['model'], tmp)
                    tmp.flush()
                    
                    with open(tmp.name, 'rb') as f:
                        ml_model.model_file.save(
                            f'{ml_model.name}.pkl',
                            ContentFile(f.read()),
                            save=False
                        )
                    
                    os.unlink(tmp.name)
                
                ml_model.save()
            
            # Определяем лучшую модель
            best_algo = automl_results['best_model']['algorithm']
            best_model = experiment.models.filter(algorithm=best_algo).first()
            experiment.best_model = best_model
            experiment.save()
            
            messages.success(request, 
                f'✅ AutoML завершен! Найдена лучшая модель: {automl_results["best_model"]["algorithm_display"]}')
            return redirect('datasets:experiment_detail', pk=experiment.pk)
        else:
            experiment.status = 'failed'
            experiment.results_summary = {'error': automl_results.get('error')}
            experiment.save()
            
            messages.error(request, f'Ошибка AutoML: {automl_results.get("error")}')
            return redirect('datasets:experiment_detail', pk=experiment.pk)
    
    return render(request, 'datasets/automl_form.html', {
        'dataset': dataset,
        'columns': get_columns(dataset)
    })


@login_required
def dataset_library(request):
    """Библиотека встроенных датасетов"""
    datasets = get_all_datasets()
    
    # Группируем по сложности
    beginner_datasets = {k: v for k, v in datasets.items() if v['difficulty'] == 'beginner'}
    intermediate_datasets = {k: v for k, v in datasets.items() if v['difficulty'] == 'intermediate'}
    
    return render(request, 'datasets/dataset_library.html', {
        'beginner_datasets': beginner_datasets,
        'intermediate_datasets': intermediate_datasets,
        'all_datasets': datasets
    })


@login_required
def import_dataset(request, dataset_id):
    """Импорт встроенного датасета"""
    try:
        dataset = import_dataset_for_user(dataset_id, request.user)
        messages.success(request, f'✅ Датасет "{dataset.name}" успешно импортирован!')
        return redirect('datasets:dataset_detail', pk=dataset.pk)
    except Exception as e:
        messages.error(request, f'Ошибка импорта: {str(e)}')
        return redirect('datasets:dataset_library')