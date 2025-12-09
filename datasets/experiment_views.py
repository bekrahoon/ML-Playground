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
        'comparison_data': comparison_data
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