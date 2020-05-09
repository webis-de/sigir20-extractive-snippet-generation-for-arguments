FROM tensorflow/tensorflow:latest-gpu-py3
RUN  pip install --upgrade setuptools
EXPOSE 5000
COPY . /usr/local/argsrank/
WORKDIR /usr/local
RUN pip install --upgrade pip
RUN pip install -r argsrank/requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/nltk_data')" ]
RUN python -m spacy download en_core_web_sm
RUN pip install ./argsrank/