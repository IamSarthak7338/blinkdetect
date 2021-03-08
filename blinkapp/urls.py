from django.urls import path
from . import views

urlpatterns = [
path('',views.home,name='home'),
    path('home',views.home,name='home'),
    path('login',views.login,name='login'),
    path('login_check',views.login_check,name='login_check'),
    path('img_from_web',views.img_from_web,name='img_from_web')
]