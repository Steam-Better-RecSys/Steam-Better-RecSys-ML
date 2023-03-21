FROM python:3.8-slim

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 80

CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "80"]