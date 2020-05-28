FROM tensorflow/tensorflow:latest-gpu-py3

ARG CONTAINER_USER
ARG CONTAINER_UID
ARG HOME_DIR=/home/${CONTAINER_USER}

ENV CONTAINER_USER=${CONTAINER_USER}
ENV CONTAINER_UID=${CONTAINER_UID}
ENV HOME_DIR=${HOME_DIR}

RUN mkdir -p ${HOME_DIR}


RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN  pip install --upgrade setuptools
EXPOSE 5000
COPY . ${HOME_DIR}/argsrank/

WORKDIR ${HOME_DIR}

RUN pip install --upgrade pip
RUN pip install -r argsrank/requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/nltk_data')" ]
RUN python -m spacy download en_core_web_sm
RUN pip install ./argsrank/