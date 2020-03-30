import os

from flask import Flask, url_for
from flask import json
from flask import request
from flask import Response
from flask import g

from argsrank import *

app = Flask(__name__)

script_dir = os.path.dirname(__file__)
stored_snippets = open(os.path.join(script_dir, "../data/snippets.txt"))
snippet_gen_app = ArgsRank()

@app.route('/')
def api_root():
    return 'welcome'

@app.route('/snippets', methods = ['POST'])
def api_get_snippets():
    json_arguments = request.json['arguments']

    cluster = []
    for argument in json_arguments:
        arg = Argument()
        arg.premises= [{"text": argument["text"]}]
        arg.id = argument["id"]
        arg.set_sentences(argument["text"])
        cluster.append(arg)

    snippets = snippet_gen_app.generate_snippet(stored_snippets, cluster)

    js = json.dumps(snippets)
    resp = Response(js, status=200, mimetype='application/json')

    return resp


def init_app():
    script_dir = os.path.dirname(__file__)

    with app.app_context():

        stored_snippets = open(os.path.join(script_dir, "../data/snippets.txt"))
        snippet_gen_app = ArgsRank()

        g.stored_snippets = stored_snippets
        g.snippet_gen_app = snippet_gen_app

#curl -H "Content-Type: application/json" --data '{"arguments":[{"id": 123, "text": "text"}]}' http://127.0.0.1:5000/snippets
if __name__ == '__main__':
    #init_app()
    app.run(debug=True)