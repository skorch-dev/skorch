FROM nvidia/cuda:8.0-cudnn7-devel

RUN apt-get update && \
	apt-get install -y python3.5-dev vim git g++ sudo zip python3-setuptools
RUN easy_install3 --upgrade pip setuptools

ENV PIP_CACHE_DIR=/cache PYTHONDONTWRITEBYTECODE=1

RUN pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl

RUN pip3 install torchvision

WORKDIR /app

ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

ADD requirements-dev.txt requirements-dev.txt
RUN pip3 install -r requirements-dev.txt

ADD . /app
RUN pip3 install -e .

# For seq2seq translation example (translation.ipynb)
ADD http://www.manythings.org/anki/fra-eng.zip /app/examples/translation/data/

RUN unzip /app/examples/translation/data/fra-eng.zip -d /app/examples/translation/data/
RUN mv /app/examples/translation/data/fra.txt /app/examples/translation/data/eng-fra.txt

# Expose jupyter notebook port (installed via requirements-dev.txt)
EXPOSE 8888
