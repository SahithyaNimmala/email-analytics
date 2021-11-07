
from django.contrib import admin
from django.apps import apps


# this will register all the models in the admin pannell
models = apps.get_models()
for model in models:
    try:
        admin.site.register(model)
    except admin.sites.AlreadyRegistered:
        pass