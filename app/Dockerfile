FROM python:latest
ENV DATABASE_URL postgresql://postgres:example@db:5432
ENV SERVER_URL 0.0.0.0:8000/
RUN pip install --upgrade pip
COPY . /app
WORKDIR /app/src
RUN pip install -r /app/requirements.txt
ENTRYPOINT uvicorn --host 0.0.0.0 --port 8000 server:app