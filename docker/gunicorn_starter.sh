#!/bin/sh

gunicorn -k uvicorn.workers.UvicornWorker main:app -w 2 --threads 2 -b 0.0.0.0:80