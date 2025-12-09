from django.urls import path
from . import views

app_name = 'collaboration'

urlpatterns = [
    # Главная сообщества
    path('', views.community_home, name='home'),
    
    # Модели
    path('models/', views.model_list, name='model_list'),
    path('models/<int:pk>/', views.model_detail, name='model_detail'),
    path('models/<int:model_pk>/publish/', views.publish_model, name='publish_model'),
    path('models/<int:pk>/unpublish/', views.unpublish_model, name='unpublish_model'),
    
    # Взаимодействие
    path('models/<int:pk>/like/', views.toggle_like, name='toggle_like'),
    path('models/<int:pk>/comment/', views.add_comment, name='add_comment'),
    path('models/<int:pk>/fork/', views.fork_model, name='fork_model'),
    
    # Личное
    path('my/', views.my_publications, name='my_publications'),
    
    # Leaderboard
    path('leaderboard/', views.leaderboard, name='leaderboard'),
]