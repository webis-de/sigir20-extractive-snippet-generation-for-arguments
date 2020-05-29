FROM tensorflow/tensorflow:latest-gpu-py3

ARG CONTAINER_USER
ARG CONTAINER_UID
ARG DEBIAN_FRONTEND=noninteractive
ARG HOME_DIR=/home/${CONTAINER_USER}

ENV CONTAINER_USER=${CONTAINER_USER}
ENV CONTAINER_UID=${CONTAINER_UID}
ENV HOME_DIR=${HOME_DIR}


RUN mkdir -p ${HOME_DIR}
RUN chown -R  ${CONTAINER_USER} ${HOME_DIR}

RUN  pip install --upgrade setuptools
EXPOSE 5000
COPY . ${HOME_DIR}/argsrank/

WORKDIR ${HOME_DIR}

RUN pip install --upgrade pip
RUN pip install -r argsrank/requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/nltk_data')" ]
RUN python -m spacy download en_core_web_sm
RUN pip install ./argsrank/