FROM python:3.8-slim

COPY ./requirements.txt /app/requirements.txt

COPY ./Data /app/Data

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

WORKDIR /app/app

RUN python model.py

EXPOSE 80

CMD ["uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "80"]