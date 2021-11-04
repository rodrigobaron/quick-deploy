FROM python:3.9

RUN apt-get update -y && apt-get install -y python3 python3-pip
RUN pip3 install --no-cache-dir fastapi gunicorn uvicorn

COPY requirements.txt /
RUN pip3 install -r /requirements.txt


COPY app /app
WORKDIR /app

COPY docker/gunicorn_starter.sh /

ENTRYPOINT ["/gunicorn_starter.sh"]