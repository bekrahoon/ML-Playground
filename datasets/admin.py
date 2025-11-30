from django.contrib import admin
from .models import Dataset, MLModel


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ['name', 'owner', 'rows_count', 'columns_count', 'created_at']
    list_filter = ['created_at', 'owner']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at', 'updated_at', 'rows_count', 'columns_count']


@admin.register(MLModel)
class MLModelAdmin(admin.ModelAdmin):
    list_display = ['name', 'algorithm', 'dataset', 'owner', 'accuracy', 'r2_score', 'created_at']
    list_filter = ['algorithm', 'created_at', 'owner']
    search_fields = ['name', 'description']
    readonly_fields = ['created_at']