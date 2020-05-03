FROM tensorflow/tensorflow:latest-gpu-py3
RUN  pip install --upgrade setuptools
COPY . /usr/local/argsrank/
WORKDIR /usr/local/argsrank
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('punkt', download_dir='/nltk_data')" ]
RUN python -m spacy download en_core_web_sm
EXPOSE 5000
RUN CUDA_VISIBLE_DEVICES=0 python src/server.py