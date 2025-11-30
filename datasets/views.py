import os
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden, FileResponse, HttpResponse
from django.core.files.base import ContentFile
from .models import Dataset, MLModel
from .forms import DatasetForm, MLModelForm
from .ml_utils import load_dataset, get_dataset_info, train_ml_model
from .export_utils import generate_pdf_report, generate_excel_report


# ============== DATASET VIEWS ==============

@login_required
def dataset_list(request):
    """Список всех датасетов пользователя"""
    datasets = Dataset.objects.filter(owner=request.user)
    return render(request, 'datasets/dataset_list.html', {'datasets': datasets})


@login_required
def dataset_create(request):
    """Создание нового датасета"""
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            dataset = form.save(commit=False)
            dataset.owner = request.user
            
            # Сохраняем файл временно для анализа
            dataset.save()
            
            # Анализируем файл
            try:
                df = pd.read_csv(dataset.file.path)
                dataset.rows_count = len(df)
                dataset.columns_count = len(df.columns)
                dataset.save()
                
                messages.success(request, 'Датасет успешно загружен!')
                return redirect('datasets:dataset_detail', pk=dataset.pk)
            except Exception as e:
                dataset.delete()
                messages.error(request, f'Ошибка при обработке файла: {str(e)}')
    else:
        form = DatasetForm()
    
    return render(request, 'datasets/dataset_form.html', {'form': form})


@login_required
def dataset_detail(request, pk):
    """Детальная информация о датасете"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    # Загружаем информацию о датасете
    df, error = load_dataset(dataset.file.path)
    
    context = {
        'dataset': dataset,
        'error': error,
    }
    
    if df is not None:
        context['info'] = get_dataset_info(df)
        context['preview'] = df.head(10).to_html(classes='table table-striped table-sm', index=False)
    
    return render(request, 'datasets/dataset_detail.html', context)


@login_required
def dataset_update(request, pk):
    """Обновление датасета"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES, instance=dataset)
        if form.is_valid():
            # Если загружен новый файл, анализируем его
            if 'file' in request.FILES:
                dataset = form.save()
                try:
                    df = pd.read_csv(dataset.file.path)
                    dataset.rows_count = len(df)
                    dataset.columns_count = len(df.columns)
                    dataset.save()
                except Exception as e:
                    messages.error(request, f'Ошибка при обработке файла: {str(e)}')
                    return redirect('datasets:dataset_update', pk=pk)
            else:
                dataset = form.save()
            
            messages.success(request, 'Датасет успешно обновлен!')
            return redirect('datasets:dataset_detail', pk=dataset.pk)
    else:
        form = DatasetForm(instance=dataset)
    
    return render(request, 'datasets/dataset_form.html', {'form': form, 'dataset': dataset})


@login_required
def dataset_delete(request, pk):
    """Удаление датасета"""
    dataset = get_object_or_404(Dataset, pk=pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    if request.method == 'POST':
        dataset.delete()
        messages.success(request, 'Датасет успешно удален!')
        return redirect('datasets:dataset_list')
    
    return render(request, 'datasets/dataset_confirm_delete.html', {'dataset': dataset})