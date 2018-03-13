FROM nvidia/cuda:8.0-cudnn7-devel

RUN apt-get update && \
	apt-get install -y python3.5-dev vim git g++ sudo zip python3-setuptools
RUN easy_install3 --upgrade pip setuptools

ENV PIP_CACHE_DIR=/cache PYTHONDONTWRITEBYTECODE=1

RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
RUN pip3 install torchvision

WORKDIR /app

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD . /app
RUN pip3 install -e .
