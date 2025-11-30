from django import forms
from .models import Dataset, MLModel


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
            # Динамически заполняем список доступных колонок
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