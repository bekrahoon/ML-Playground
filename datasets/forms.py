from django import forms
from .models import Dataset, MLModel, Experiment


class DatasetForm(forms.ModelForm):
    class Meta:
        model = Dataset
        fields = ['name', 'description', 'file']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Название датасета'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Описание датасета'
            }),
            'file': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': '.csv'
            })
        }


class MLModelForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['name', 'description', 'algorithm', 'target_column']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Название модели'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Описание модели'
            }),
            'algorithm': forms.Select(attrs={
                'class': 'form-select'
            }),
            'target_column': forms.Select(attrs={
                'class': 'form-select'
            })
        }
    
    def __init__(self, *args, dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        if dataset:
            import pandas as pd
            try:
                df = pd.read_csv(dataset.file.path)
                columns = [(col, col) for col in df.columns]
                self.fields['target_column'].widget = forms.Select(
                    choices=columns,
                    attrs={'class': 'form-select'}
                )
            except Exception:
                pass


class ExperimentForm(forms.ModelForm):
    """Форма для создания ML эксперимента"""
    
    ALGORITHM_CHOICES = [
        ('linear_regression', 'Linear Regression'),
        ('logistic_regression', 'Logistic Regression'),
        ('decision_tree_classifier', 'Decision Tree Classifier'),
        ('decision_tree_regressor', 'Decision Tree Regressor'),
        ('random_forest_classifier', 'Random Forest Classifier'),
        ('random_forest_regressor', 'Random Forest Regressor'),
    ]
    
    algorithms = forms.MultipleChoiceField(
        choices=ALGORITHM_CHOICES,
        widget=forms.CheckboxSelectMultiple(attrs={
            'class': 'form-check-input'
        }),
        label='Выберите алгоритмы для сравнения'
    )
    
    class Meta:
        model = Experiment
        fields = ['name', 'description', 'target_column', 'test_size']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Название эксперимента'
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': 'Описание эксперимента'
            }),
            'target_column': forms.Select(attrs={
                'class': 'form-select'
            }),
            'test_size': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': '0.1',
                'max': '0.5',
                'step': '0.05',
                'value': '0.2'
            })
        }
    
    def __init__(self, *args, dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        if dataset:
            import pandas as pd
            try:
                df = pd.read_csv(dataset.file.path)
                columns = [(col, col) for col in df.columns]
                self.fields['target_column'].widget = forms.Select(
                    choices=columns,
                    attrs={'class': 'form-select'}
                )
            except Exception:
                pass