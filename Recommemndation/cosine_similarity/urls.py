from django.urls import path
from . import views


urlpatterns = [
    path('<str:ingredient>/', views.get_similar_recipes)
]