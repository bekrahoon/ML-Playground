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

# ============== ML MODEL VIEWS ==============

@login_required
def model_list(request):
    """Список всех моделей пользователя"""
    models = MLModel.objects.filter(owner=request.user)
    return render(request, 'datasets/model_list.html', {'models': models})


@login_required
def model_create(request, dataset_pk):
    """Создание и обучение новой ML модели"""
    dataset = get_object_or_404(Dataset, pk=dataset_pk)
    
    # Проверка доступа
    if dataset.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этому датасету')
    
    if request.method == 'POST':
        form = MLModelForm(request.POST, dataset=dataset)
        
        if form.is_valid():
            # Получаем выбранные признаки
            feature_columns = request.POST.getlist('feature_columns')
            
            if not feature_columns:
                messages.error(request, 'Выберите хотя бы один признак!')
                return render(request, 'datasets/model_form.html', {
                    'form': form,
                    'dataset': dataset,
                    'columns': get_columns(dataset)
                })
            
            # Обучаем модель
            model_obj, metrics, error = train_ml_model(
                dataset.file.path,
                form.cleaned_data['algorithm'],
                form.cleaned_data['target_column'],
                feature_columns
            )
            
            if error:
                messages.error(request, error)
                return render(request, 'datasets/model_form.html', {
                    'form': form,
                    'dataset': dataset,
                    'columns': get_columns(dataset)
                })
            
            # Сохраняем модель
            ml_model = form.save(commit=False)
            ml_model.dataset = dataset
            ml_model.owner = request.user
            ml_model.feature_columns = feature_columns
            
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
            import pickle
            import tempfile
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
            
            messages.success(request, 'Модель успешно обучена!')
            return redirect('datasets:model_detail', pk=ml_model.pk)
    else:
        form = MLModelForm(dataset=dataset)
    
    return render(request, 'datasets/model_form.html', {
        'form': form,
        'dataset': dataset,
        'columns': get_columns(dataset)
    })


@login_required
def model_detail(request, pk):
    """Детальная информация о модели"""
    model = get_object_or_404(MLModel, pk=pk)
    
    # Проверка доступа
    if model.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    return render(request, 'datasets/model_detail.html', {'model': model})

@login_required
def model_delete(request, pk):
    """Удаление модели"""
    model = get_object_or_404(MLModel, pk=pk)
    
    # Проверка доступа
    if model.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    if request.method == 'POST':
        model.delete()
        messages.success(request, 'Модель успешно удалена!')
        return redirect('datasets:model_list')
    
    return render(request, 'datasets/model_confirm_delete.html', {'model': model})


# ============== EXPORT VIEWS ==============

@login_required
def export_pdf(request, pk):
    """Экспорт модели в PDF"""
    model = get_object_or_404(MLModel, pk=pk)
    
    # Проверка доступа
    if model.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    buffer = generate_pdf_report(model)
    
    response = FileResponse(buffer, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{model.name}_report.pdf"'
    return response


@login_required
def export_excel(request, pk):
    """Экспорт модели в Excel"""
    model = get_object_or_404(MLModel, pk=pk)
    
    # Проверка доступа
    if model.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    buffer = generate_excel_report(model)
    
    response = HttpResponse(
        buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename="{model.name}_report.xlsx"'
    return response


# ============== HELPER FUNCTIONS ==============

def get_columns(dataset):
    """Получение списка колонок датасета"""
    try:
        df = pd.read_csv(dataset.file.path)
        return list(df.columns)
    except Exception:
        return []