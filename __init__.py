import os

from flask import Flask
from flask import json
from flask import request
from flask import Response

from argsrank.lib import argsrank
from argsrank.lib.argument import Argument

from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})


def load_global_data():
    global snippet_gen_app
    global stored_snippets

    snippet_gen_app = argsrank.ArgsRank()

    script_dir = os.path.dirname(__file__)
    stored_snippets = json.load(open(os.path.join(script_dir, "./data/snippets.txt")))

def create_app(test_config=None):

    load_global_data()

    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'argsrank.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass



    @app.before_request
    def log_the_request():
        app.logger.info('Headers: %s', request.headers)

    @app.route('/')
    def api_root():
        return 'welcome'

    @app.route('/snippets', methods = ['POST'])
    def api_get_snippets():
        json_arguments = json.loads(request.data, encoding='latin1')
        json_arguments = json_arguments['arguments']

        cluster = []
        for argument in json_arguments:
            arg = Argument()
            arg.premises= [{"text": argument["text"]}]
            arg.id = argument["id"]
            arg.set_sentences(argument["text"])
            cluster.append(arg)

        #log argument ids
        app.logger.info('Body: %s', '\t'.join([arg.id for arg in cluster]))

        snippets = snippet_gen_app.generate_snippet(stored_snippets, cluster)

        js = json.dumps(snippets)
        resp = Response(js, status=200, mimetype='application/json')

        return resp


    return app