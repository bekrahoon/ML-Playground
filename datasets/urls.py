from django.urls import path
from . import views
from . import experiment_views

app_name = 'datasets'

urlpatterns = [
    # Dataset URLs
    path('', views.dataset_list, name='dataset_list'),
    path('create/', views.dataset_create, name='dataset_create'),
    path('library/', experiment_views.dataset_library, name='dataset_library'),
    path('library/import/<str:dataset_id>/', experiment_views.import_dataset, name='import_dataset'),
    path('<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('<int:pk>/update/', views.dataset_update, name='dataset_update'),
    path('<int:pk>/delete/', views.dataset_delete, name='dataset_delete'),
    
    # ML Model URLs
    path('models/', views.model_list, name='model_list'),
    path('<int:dataset_pk>/train/', views.model_create, name='model_create'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:pk>/delete/', views.model_delete, name='model_delete'),
    
    # Export URLs
    path('models/<int:pk>/export/pdf/', views.export_pdf, name='export_pdf'),
    path('models/<int:pk>/export/excel/', views.export_excel, name='export_excel'),
    
    # Experiment URLs
    path('experiments/', experiment_views.experiment_list, name='experiment_list'),
    path('<int:dataset_pk>/experiment/create/', experiment_views.experiment_create, name='experiment_create'),
    path('<int:dataset_pk>/automl/create/', experiment_views.automl_create, name='automl_create'),
    path('experiments/<int:pk>/', experiment_views.experiment_detail, name='experiment_detail'),
    path('experiments/<int:pk>/delete/', experiment_views.experiment_delete, name='experiment_delete'),
    path('experiments/<int:pk>/compare/json/', experiment_views.experiment_compare_json, name='experiment_compare_json'),
]