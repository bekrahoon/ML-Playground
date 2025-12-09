from django.urls import path
from . import views

app_name = 'datasets'

urlpatterns = [
    # Dataset URLs
    path('', views.dataset_list, name='dataset_list'),
    path('create/', views.dataset_create, name='dataset_create'),
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
]