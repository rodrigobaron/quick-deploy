FROM nvcr.io/nvidia/tritonserver:21.11-py3

ADD ./ /fastdeploy
RUN pip install /fastdeploy
RUN pip install -r /fastdeploy/requirements.txt

ENTRYPOINT ["/fastdeploy/docker/wrapper.sh"]