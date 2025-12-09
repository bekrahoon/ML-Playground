from django.db import models
from django.contrib.auth.models import User
from datasets.models import MLModel, Dataset


class PublicModel(models.Model):
    """Публичная модель в сообществе"""
    
    VISIBILITY_CHOICES = [
        ('public', 'Публичная'),
        ('unlisted', 'По ссылке'),
        ('private', 'Приватная'),
    ]
    
    # Связь с оригинальной моделью
    original_model = models.OneToOneField(
        MLModel, 
        on_delete=models.CASCADE, 
        related_name='public_model'
    )
    
    # Информация о публикации
    title = models.CharField('Название', max_length=200)
    description = models.TextField('Описание')
    visibility = models.CharField('Видимость', max_length=20, choices=VISIBILITY_CHOICES, default='public')
    
    # Теги
    tags = models.CharField('Теги', max_length=500, help_text='Через запятую: классификация, медицина, точность>90%')
    
    # Use cases
    use_cases = models.TextField('Примеры использования', blank=True)
    
    # Метаданные
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='public_models')
    created_at = models.DateTimeField('Дата публикации', auto_now_add=True)
    updated_at = models.DateTimeField('Дата обновления', auto_now=True)
    
    # Статистика
    views_count = models.IntegerField('Просмотры', default=0)
    downloads_count = models.IntegerField('Скачивания', default=0)
    forks_count = models.IntegerField('Форки', default=0)
    
    # Рейтинг
    likes_count = models.IntegerField('Лайки', default=0)
    
    # Featured
    is_featured = models.BooleanField('Рекомендуемая', default=False)
    featured_at = models.DateTimeField('Дата рекомендации', null=True, blank=True)
    
    class Meta:
        verbose_name = 'Публичная модель'
        verbose_name_plural = 'Публичные модели'
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['-likes_count']),
            models.Index(fields=['-views_count']),
        ]
    
    def __str__(self):
        return f'{self.title} by {self.author.username}'
    
    def get_tags_list(self):
        """Возвращает список тегов"""
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]
    
    def increment_views(self):
        """Увеличить счетчик просмотров"""
        self.views_count += 1
        self.save(update_fields=['views_count'])
    
    def increment_downloads(self):
        """Увеличить счетчик скачиваний"""
        self.downloads_count += 1
        self.save(update_fields=['downloads_count'])


class ModelLike(models.Model):
    """Лайк модели"""
    
    public_model = models.ForeignKey(PublicModel, on_delete=models.CASCADE, related_name='likes')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='model_likes')
    created_at = models.DateTimeField('Дата', auto_now_add=True)
    
    class Meta:
        verbose_name = 'Лайк'
        verbose_name_plural = 'Лайки'
        unique_together = ['public_model', 'user']
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.user.username} likes {self.public_model.title}'


class ModelComment(models.Model):
    """Комментарий к модели"""
    
    public_model = models.ForeignKey(PublicModel, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='model_comments')
    
    text = models.TextField('Текст комментария')
    
    # Вложенные комментарии
    parent = models.ForeignKey(
        'self', 
        null=True, 
        blank=True, 
        on_delete=models.CASCADE, 
        related_name='replies'
    )
    
    created_at = models.DateTimeField('Дата', auto_now_add=True)
    updated_at = models.DateTimeField('Дата изменения', auto_now=True)
    
    # Модерация
    is_edited = models.BooleanField('Отредактирован', default=False)
    is_deleted = models.BooleanField('Удален', default=False)
    
    class Meta:
        verbose_name = 'Комментарий'
        verbose_name_plural = 'Комментарии'
        ordering = ['created_at']
    
    def __str__(self):
        return f'Comment by {self.author.username} on {self.public_model.title}'


class ModelFork(models.Model):
    """Форк модели (копирование в свой аккаунт)"""
    
    original_public_model = models.ForeignKey(
        PublicModel, 
        on_delete=models.CASCADE, 
        related_name='forks'
    )
    
    forked_model = models.ForeignKey(
        MLModel, 
        on_delete=models.CASCADE, 
        related_name='forked_from'
    )
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='model_forks')
    created_at = models.DateTimeField('Дата форка', auto_now_add=True)
    
    class Meta:
        verbose_name = 'Форк модели'
        verbose_name_plural = 'Форки моделей'
        ordering = ['-created_at']
        unique_together = ['original_public_model', 'user']
    
    def __str__(self):
        return f'{self.user.username} forked {self.original_public_model.title}'


class PublicDataset(models.Model):
    """Публичный датасет в сообществе"""
    
    VISIBILITY_CHOICES = [
        ('public', 'Публичный'),
        ('unlisted', 'По ссылке'),
        ('private', 'Приватный'),
    ]
    
    # Связь с оригинальным датасетом
    original_dataset = models.OneToOneField(
        Dataset, 
        on_delete=models.CASCADE, 
        related_name='public_dataset'
    )
    
    # Информация
    title = models.CharField('Название', max_length=200)
    description = models.TextField('Описание')
    visibility = models.CharField('Видимость', max_length=20, choices=VISIBILITY_CHOICES, default='public')
    
    # Теги
    tags = models.CharField('Теги', max_length=500)
    
    # Метаданные
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='public_datasets')
    created_at = models.DateTimeField('Дата публикации', auto_now_add=True)
    
    # Статистика
    views_count = models.IntegerField('Просмотры', default=0)
    downloads_count = models.IntegerField('Скачивания', default=0)
    
    class Meta:
        verbose_name = 'Публичный датасет'
        verbose_name_plural = 'Публичные датасеты'
        ordering = ['-created_at']
    
    def __str__(self):
        return f'{self.title} by {self.author.username}'
    
    def get_tags_list(self):
        return [tag.strip() for tag in self.tags.split(',') if tag.strip()]