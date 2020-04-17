FROM tensorflow/tensorflow:latest-gpu
COPY . /usr/local/argsrank/
WORKDIR /usr/local/argsrank
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/nltk_data')" ]
RUN python -m spacy download en_core_web_sm