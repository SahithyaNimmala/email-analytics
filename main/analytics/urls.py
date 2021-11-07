from django.urls import path
from .views import HomeView, InsightView, MailView, PredictView

urlpatterns = [
    path("home", HomeView.as_view(), name="Home"),
    path('message/<str:thread_id>/', MailView.as_view(),name='mail'),
    path('predict',PredictView.as_view(),name='predict'),
    path('insight',InsightView.as_view(),name='insight')
]