FROM nvcr.io/nvidia/tritonserver:21.11-py3

ADD ./ /quick-deploy
RUN pip install /quick-deploy
RUN pip install -r /quick-deploy/requirements.txt

ENTRYPOINT ["/quick-deploy/docker/wrapper.sh"]