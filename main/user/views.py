from .models import User
from django.views import View
from django.contrib.auth.views import LoginView


class CustomLoginView(LoginView):
    template_name = "user/login.html"
    redirect_authenticated_user = True

