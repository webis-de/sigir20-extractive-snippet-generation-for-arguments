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
RUN CUDA_VISIBLE_DEVICES=5 waitress-serve --listen=0.0.0.0:5000  --call 'argsrank:create_app'


#RUN CUDA_VISIBLE_DEVICES=0 python src/server.py