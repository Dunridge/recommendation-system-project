from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np
import movielens_sequence.movielens_sequence as ms

app = Flask(__name__)
api = Api(app)

CORS(app)


@app.route('/', methods=['GET'])
def get():
    return {'hello': 'world'}


@app.route('/number', methods=['GET'])
def number():
    return {'number': 5}


@app.route('/lstm', methods=['GET'])
def get_lstm_results():
    return {'results': ms.run_lstm_model()}


if __name__ == '__main__':
    #ms.run_lstm_model()
    app.run(debug=True)
