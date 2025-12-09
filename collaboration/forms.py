from django import forms
from .models import PublicModel, ModelComment, PublicDataset


class PublishModelForm(forms.ModelForm):
    """Форма для публикации модели"""
    
    title = forms.CharField(
        label='Название',
        max_length=200,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Например: Модель для предсказания оттока клиентов'
        })
    )
    
    description = forms.CharField(
        label='Описание',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Опишите вашу модель: что она делает, на каких данных обучена, какие метрики...'
        })
    )
    
    tags = forms.CharField(
        label='Теги',
        max_length=500,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'классификация, медицина, точность>90%, random-forest'
        }),
        help_text='Теги через запятую для поиска'
    )
    
    use_cases = forms.CharField(
        label='Примеры использования',
        required=False,
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Где можно использовать эту модель?'
        })
    )
    
    visibility = forms.ChoiceField(
        label='Видимость',
        choices=PublicModel.VISIBILITY_CHOICES,
        initial='public',
        widget=forms.RadioSelect()
    )
    
    class Meta:
        model = PublicModel
        fields = ['title', 'description', 'tags', 'use_cases', 'visibility']


class CommentForm(forms.ModelForm):
    """Форма для комментария"""
    
    text = forms.CharField(
        label='',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 3,
            'placeholder': 'Написать комментарий...'
        })
    )
    
    class Meta:
        model = ModelComment
        fields = ['text']


class PublishDatasetForm(forms.ModelForm):
    """Форма для публикации датасета"""
    
    title = forms.CharField(
        label='Название',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Название датасета'
        })
    )
    
    description = forms.CharField(
        label='Описание',
        widget=forms.Textarea(attrs={
            'class': 'form-control',
            'rows': 5,
            'placeholder': 'Опишите датасет...'
        })
    )
    
    tags = forms.CharField(
        label='Теги',
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'регрессия, продажи, временные-ряды'
        })
    )
    
    visibility = forms.ChoiceField(
        label='Видимость',
        choices=PublicDataset.VISIBILITY_CHOICES,
        widget=forms.RadioSelect()
    )
    
    class Meta:
        model = PublicDataset
        fields = ['title', 'description', 'tags', 'visibility']