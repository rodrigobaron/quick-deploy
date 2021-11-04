FROM python:3.8

# Install wsgi
RUN apt-get update -y \
    && apt-get install -y libapache2-mod-wsgi-py3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup app
COPY app /app
WORKDIR /app
RUN pip install -r requirements.txt
ENV FLASK_APP /app/server.py

CMD gunicorn --bind 0.0.0.0:8080 --timeout=600 wsgi:app -w 4

EXPOSE 8080