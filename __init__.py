import os

from flask import Flask
from flask import json
from flask import request
from flask import Response

from argsrank.lib import argsrank
from argsrank.lib import argument

def create_app(test_config=None):

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

    print('initializing snippet generation app..')
    argsrank.init_snippet_gen_app()

    @app.route('/')
    def api_root():
        return 'welcome'

    @app.route('/snippets', methods = ['POST'])
    def api_get_snippets():
        json_arguments = request.json['arguments']

        cluster = []
        for argument in json_arguments:
            arg = argument.Argument()
            arg.premises= [{"text": argument["text"]}]
            arg.id = argument["id"]
            arg.set_sentences(argument["text"])
            cluster.append(arg)

        print('generate snippets...')
        snippet_gen_app = argsRank.get_args_snippet_gen()
        snippets = snippet_gen_app.generate_snippet(stored_snippets, cluster)

        js = json.dumps(snippets)
        resp = Response(js, status=200, mimetype='application/json')

        return resp


    return app