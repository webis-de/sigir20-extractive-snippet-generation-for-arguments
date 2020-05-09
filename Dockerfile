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
RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8
RUN export FLASK_APP=argsrank
RUN export FLASK_ENV=production
RUN CUDA_VISIBLE_DEVICES=5 flask run
RUN waitress-serve --listen=0.0.0.0:5000  --call 'argsrank:create_app'


#RUN CUDA_VISIBLE_DEVICES=0 python src/server.py