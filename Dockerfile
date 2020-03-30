FROM tensorflow/tensorflow:devel-py3
COPY . /usr/local/argsrank/
WORKDIR /usr/local/argsrank
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt')" ]
RUN python -m spacy download en_core_web_sm