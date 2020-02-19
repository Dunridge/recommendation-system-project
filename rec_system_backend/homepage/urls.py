
from django.urls import path
from . import views

urlpatterns = [
    path('homepage/', views.index, name='index'),
]
