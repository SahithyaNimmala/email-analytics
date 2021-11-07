from django.shortcuts import redirect, render
from django.http import JsonResponse
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views import View
from .utils import get_message, get_messages
from ml_model.main import predict_message


class HomeView(LoginRequiredMixin, View):
    template_name = "main/home.html"

    def get(self, request):
        mails = get_messages(request.user)
        return render(request, self.template_name, {"mails": mails})


class MailView(LoginRequiredMixin, View):
    template_name = "main/mail_view.html"

    def get(self, request, thread_id):
        mail = get_message(request.user, thread_id)
        return render(request, self.template_name, {"mail": mail})


class PredictView(View):
    def post(self, request):
        try:
            data = request.data
            return JsonResponse({"message": predict_message(data["text"])})
        except Exception as e:
            return JsonResponse({"error": [e.__str__()]})


class InsightView(View):
    template_name = "main/insight.html"

    def get(self, request):
        limit = request.GET.get("q", 10)

        mails = get_messages(request.user, limit)
        for i in range(0, len(mails)):
            mails[i]["results"] = predict_message(mails[i]["snippet"])
        return render(request, self.template_name, {"mails": mails})
