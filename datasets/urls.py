from django.urls import path
from . import views

app_name = 'datasets'

urlpatterns = [
    path('', views.dataset_list, name='dataset_list'),
    path('create/', views.dataset_create, name='dataset_create'),
    path('<int:pk>/', views.dataset_detail, name='dataset_detail'),
    path('<int:pk>/edit/', views.dataset_update, name='dataset_update'),
    path('<int:pk>/delete/', views.dataset_delete, name='dataset_delete'),
]
