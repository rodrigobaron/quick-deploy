FROM python:3.9

ADD ./ /quick-deploy
RUN pip install /quick-deploy
RUN pip install -r /quick-deploy/requirements.txt

ENTRYPOINT ["quick-deploy"]