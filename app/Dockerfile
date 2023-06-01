FROM python:3.8-slim

RUN pip install --upgrade pip

ENV PIP_ROOT_USER_ACTION=ignore

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 3000

CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "3000", "--log-level", "critical"]