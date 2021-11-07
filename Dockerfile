FROM python:3.9-slim
ENV PYTHONUNBUFFERED 1
ENV PORT 8000
ENV PYTHONDONTWRITEBYTECODE 1
ENV DEBUG 0

WORKDIR /code
COPY requirements.txt /code/
RUN pip install -r requirements.txt
COPY main /code/
