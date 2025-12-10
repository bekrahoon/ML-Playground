from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponseForbidden, JsonResponse
from django.db.models import Q, Count
from datasets.models import MLModel, Dataset
from .models import PublicModel, ModelLike, ModelComment, ModelFork, PublicDataset
from .forms import PublishModelForm, CommentForm, PublishDatasetForm
import pickle
import tempfile
import os
from django.core.files.base import ContentFile
from django.db import models



def community_home(request):
    """Главная страница сообщества"""
    # Топ модели
    featured_models = PublicModel.objects.filter(
        visibility='public',
        is_featured=True
    )[:3]
    
    # Популярные модели
    popular_models = PublicModel.objects.filter(
        visibility='public'
    ).order_by('-likes_count', '-views_count')[:6]
    
    # Последние модели
    recent_models = PublicModel.objects.filter(
        visibility='public'
    ).order_by('-created_at')[:6]
    
    # Статистика
    stats = {
        'total_models': PublicModel.objects.filter(visibility='public').count(),
        'total_users': PublicModel.objects.values('author').distinct().count(),
        'total_likes': ModelLike.objects.count(),
        'total_comments': ModelComment.objects.filter(is_deleted=False).count(),
    }
    
    return render(request, 'collaboration/home.html', {
        'featured_models': featured_models,
        'popular_models': popular_models,
        'recent_models': recent_models,
        'stats': stats,
    })


def model_list(request):
    """Список всех публичных моделей"""
    models = PublicModel.objects.filter(visibility='public')
    
    # Поиск
    search_query = request.GET.get('q', '')
    if search_query:
        models = models.filter(
            Q(title__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(tags__icontains=search_query)
        )
    
    # Фильтр по тегу
    tag = request.GET.get('tag', '')
    if tag:
        models = models.filter(tags__icontains=tag)
    
    # Сортировка
    sort_by = request.GET.get('sort', 'recent')
    if sort_by == 'popular':
        models = models.order_by('-likes_count', '-views_count')
    elif sort_by == 'downloads':
        models = models.order_by('-downloads_count')
    else:  # recent
        models = models.order_by('-created_at')
    
    # Собираем все теги
    all_tags = set()
    for model in PublicModel.objects.filter(visibility='public'):
        all_tags.update(model.get_tags_list())
    
    return render(request, 'collaboration/model_list.html', {
        'models': models,
        'search_query': search_query,
        'current_tag': tag,
        'current_sort': sort_by,
        'all_tags': sorted(all_tags),
    })


def model_detail(request, pk):
    """Детальная страница публичной модели"""
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    # Проверка доступа
    if public_model.visibility == 'private':
        if not request.user.is_authenticated or request.user != public_model.author:
            return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    # Увеличиваем счетчик просмотров
    if not request.user.is_authenticated or request.user != public_model.author:
        public_model.increment_views()
    
    # Проверяем лайк текущего пользователя
    user_liked = False
    if request.user.is_authenticated:
        user_liked = ModelLike.objects.filter(
            public_model=public_model,
            user=request.user
        ).exists()
    
    # Комментарии
    comments = public_model.comments.filter(
        is_deleted=False,
        parent=None
    ).select_related('author')
    
    # Форма комментария
    comment_form = CommentForm()
    
    return render(request, 'collaboration/model_detail.html', {
        'public_model': public_model,
        'original_model': public_model.original_model,
        'user_liked': user_liked,
        'comments': comments,
        'comment_form': comment_form,
    })


@login_required
def publish_model(request, model_pk):
    """Опубликовать модель в сообщество"""
    original_model = get_object_or_404(MLModel, pk=model_pk)
    
    # Проверка доступа
    if original_model.owner != request.user:
        return HttpResponseForbidden('У вас нет доступа к этой модели')
    
    # Проверка - уже опубликована?
    if hasattr(original_model, 'public_model'):
        messages.info(request, 'Эта модель уже опубликована')
        return redirect('collaboration:model_detail', pk=original_model.public_model.pk)
    
    if request.method == 'POST':
        form = PublishModelForm(request.POST)
        if form.is_valid():
            public_model = form.save(commit=False)
            public_model.original_model = original_model
            public_model.author = request.user
            public_model.save()
            
            messages.success(request, f'✅ Модель "{public_model.title}" успешно опубликована!')
            return redirect('collaboration:model_detail', pk=public_model.pk)
    else:
        # Предзаполнение формы
        initial_data = {
            'title': original_model.name,
            'description': original_model.description or f'Модель {original_model.get_algorithm_display()} для {original_model.dataset.name}',
            'tags': f'{original_model.algorithm}, {original_model.dataset.name}'
        }
        form = PublishModelForm(initial=initial_data)
    
    return render(request, 'collaboration/publish_model.html', {
        'form': form,
        'original_model': original_model,
    })


@login_required
def toggle_like(request, pk):
    """Лайк/анлайк модели"""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=400)
    
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    like, created = ModelLike.objects.get_or_create(
        public_model=public_model,
        user=request.user
    )
    
    if not created:
        # Убрать лайк
        like.delete()
        public_model.likes_count -= 1
        liked = False
    else:
        # Добавить лайк
        public_model.likes_count += 1
        liked = True
    
    public_model.save(update_fields=['likes_count'])
    
    return JsonResponse({
        'liked': liked,
        'likes_count': public_model.likes_count
    })

@login_required
def add_comment(request, pk):
    """Добавить комментарий"""
    if request.method != 'POST':
        return redirect('collaboration:model_detail', pk=pk)
    
    public_model = get_object_or_404(PublicModel, pk=pk)
    form = CommentForm(request.POST)
    
    if form.is_valid():
        comment = form.save(commit=False)
        comment.public_model = public_model
        comment.author = request.user
        
        # Проверка на вложенный комментарий
        parent_id = request.POST.get('parent_id')
        if parent_id:
            comment.parent = get_object_or_404(ModelComment, pk=parent_id)
        
        comment.save()
        messages.success(request, 'Комментарий добавлен')
    
    return redirect('collaboration:model_detail', pk=pk)


@login_required
def fork_model(request, pk):
    """Форк модели (скопировать себе)"""
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    # Проверка - уже есть форк?
    existing_fork = ModelFork.objects.filter(
        original_public_model=public_model,
        user=request.user
    ).first()
    
    if existing_fork:
        messages.info(request, 'Вы уже сделали форк этой модели')
        return redirect('datasets:model_detail', pk=existing_fork.forked_model.pk)
    
    try:
        original = public_model.original_model
        
        # Загружаем pickle модели
        with original.model_file.open('rb') as f:
            model_obj = pickle.load(f)
        
        # Создаем новую модель
        new_model = MLModel(
            name=f'{public_model.title} (fork)',
            description=f'Форк модели от {public_model.author.username}. {public_model.description}',
            dataset=original.dataset,
            algorithm=original.algorithm,
            target_column=original.target_column,
            feature_columns=original.feature_columns,
            owner=request.user,
            
            # Копируем метрики
            accuracy=original.accuracy,
            f1_score=original.f1_score,
            mse=original.mse,
            rmse=original.rmse,
            r2_score=original.r2_score,
            confusion_matrix=original.confusion_matrix,
            training_time=original.training_time,
        )
        
        # Сохраняем pickle файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(model_obj, tmp)
            tmp.flush()
            
            with open(tmp.name, 'rb') as f:
                new_model.model_file.save(
                    f'{new_model.name}.pkl',
                    ContentFile(f.read()),
                    save=False
                )
            
            os.unlink(tmp.name)
        
        new_model.save()
        
        # Создаем запись о форке
        fork = ModelFork.objects.create(
            original_public_model=public_model,
            forked_model=new_model,
            user=request.user
        )
        
        # Увеличиваем счетчик форков
        public_model.forks_count += 1
        public_model.save(update_fields=['forks_count'])
        
        messages.success(request, f'✅ Модель "{public_model.title}" успешно скопирована!')
        return redirect('datasets:model_detail', pk=new_model.pk)
        
    except Exception as e:
        messages.error(request, f'Ошибка при форке модели: {str(e)}')
        return redirect('collaboration:model_detail', pk=pk)


@login_required
def unpublish_model(request, pk):
    """Снять модель с публикации"""
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    # Проверка доступа
    if public_model.author != request.user:
        return HttpResponseForbidden('У вас нет доступа')
    
    if request.method == 'POST':
        model_title = public_model.title
        public_model.delete()
        messages.success(request, f'Модель "{model_title}" снята с публикации')
        return redirect('collaboration:my_publications')
    
    return render(request, 'collaboration/unpublish_confirm.html', {
        'public_model': public_model
    })


@login_required
def my_publications(request):
    """Мои опубликованные модели"""
    my_models = PublicModel.objects.filter(author=request.user)
    
    return render(request, 'collaboration/my_publications.html', {
        'my_models': my_models
    })


def leaderboard(request):
    """Таблица лидеров"""
    # Топ авторов по лайкам
    top_authors = PublicModel.objects.filter(
        visibility='public'
    ).values('author__username', 'author__id').annotate(
        total_likes=Count('likes'),
        total_models=Count('id'),
        total_views=models.Sum('views_count')
    ).order_by('-total_likes')[:20]
    
    # Топ модели
    top_models = PublicModel.objects.filter(
        visibility='public'
    ).order_by('-likes_count')[:10]
    
    return render(request, 'collaboration/leaderboard.html', {
        'top_authors': top_authors,
        'top_models': top_models
    })

@login_required
def add_comment(request, pk):
    """Добавить комментарий"""
    if request.method != 'POST':
        return redirect('collaboration:model_detail', pk=pk)
    
    public_model = get_object_or_404(PublicModel, pk=pk)
    form = CommentForm(request.POST)
    
    if form.is_valid():
        comment = form.save(commit=False)
        comment.public_model = public_model
        comment.author = request.user
        
        # Проверка на вложенный комментарий
        parent_id = request.POST.get('parent_id')
        if parent_id:
            comment.parent = get_object_or_404(ModelComment, pk=parent_id)
        
        comment.save()
        messages.success(request, 'Комментарий добавлен')
    
    return redirect('collaboration:model_detail', pk=pk)


@login_required
def fork_model(request, pk):
    """Форк модели (скопировать себе)"""
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    # Проверка - уже есть форк?
    existing_fork = ModelFork.objects.filter(
        original_public_model=public_model,
        user=request.user
    ).first()
    
    if existing_fork:
        messages.info(request, 'Вы уже сделали форк этой модели')
        return redirect('datasets:model_detail', pk=existing_fork.forked_model.pk)
    
    try:
        original = public_model.original_model
        
        # Загружаем pickle модели
        with original.model_file.open('rb') as f:
            model_obj = pickle.load(f)
        
        # Создаем новую модель
        new_model = MLModel(
            name=f'{public_model.title} (fork)',
            description=f'Форк модели от {public_model.author.username}. {public_model.description}',
            dataset=original.dataset,
            algorithm=original.algorithm,
            target_column=original.target_column,
            feature_columns=original.feature_columns,
            owner=request.user,
            
            # Копируем метрики
            accuracy=original.accuracy,
            f1_score=original.f1_score,
            mse=original.mse,
            rmse=original.rmse,
            r2_score=original.r2_score,
            confusion_matrix=original.confusion_matrix,
            training_time=original.training_time,
        )
        
        # Сохраняем pickle файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
            pickle.dump(model_obj, tmp)
            tmp.flush()
            
            with open(tmp.name, 'rb') as f:
                new_model.model_file.save(
                    f'{new_model.name}.pkl',
                    ContentFile(f.read()),
                    save=False
                )
            
            os.unlink(tmp.name)
        
        new_model.save()
        
        # Создаем запись о форке
        fork = ModelFork.objects.create(
            original_public_model=public_model,
            forked_model=new_model,
            user=request.user
        )
        
        # Увеличиваем счетчик форков
        public_model.forks_count += 1
        public_model.downloads_count += 1
        public_model.save()
        
        messages.success(request, f'✅ Модель "{public_model.title}" успешно скопирована!')
        return redirect('datasets:model_detail', pk=new_model.pk)
        
    except Exception as e:
        messages.error(request, f'Ошибка при форке модели: {str(e)}')
        return redirect('collaboration:model_detail', pk=pk)


@login_required
def unpublish_model(request, pk):
    """Снять модель с публикации"""
    public_model = get_object_or_404(PublicModel, pk=pk)
    
    # Проверка доступа
    if public_model.author != request.user:
        return HttpResponseForbidden('У вас нет доступа')
    
    if request.method == 'POST':
        model_title = public_model.title
        public_model.delete()
        messages.success(request, f'Модель "{model_title}" снята с публикации')
        return redirect('collaboration:my_publications')
    
    return render(request, 'collaboration/unpublish_confirm.html', {
        'public_model': public_model
    })


@login_required
def my_publications(request):
    """Мои опубликованные модели"""
    my_models = PublicModel.objects.filter(author=request.user)
    
    return render(request, 'collaboration/my_publications.html', {
        'my_models': my_models
    })


def leaderboard(request):
    """Таблица лидеров"""
    # Топ авторов по лайкам
    top_authors = PublicModel.objects.filter(
        visibility='public'
    ).values('author__username', 'author__id').annotate(
        total_likes=Count('likes'),
        total_models=Count('id'),
        total_views=models.Sum('views_count')
    ).order_by('-total_likes')[:20]
    
    # Топ модели
    top_models = PublicModel.objects.filter(
        visibility='public'
    ).order_by('-likes_count')[:10]
    
    return render(request, 'collaboration/leaderboard.html', {
        'top_authors': top_authors,
        'top_models': top_models
    })