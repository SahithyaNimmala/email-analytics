from django.urls import path
from .views import CustomLoginView
from django.contrib.auth import views

urlpatterns = [
    path("login", CustomLoginView.as_view(), name="login"),
    path('logout',views.LogoutView.as_view(next_page='login'),name='logout'),
]