FROM python:3.9

ADD ./ /fastdeploy
RUN pip install /fastdeploy
RUN pip install -r /fastdeploy/requirements.txt

ENTRYPOINT ["fast-deploy"]