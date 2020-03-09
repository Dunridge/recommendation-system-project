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
def get_lstm_results(): # TODO: work on from here, the error was caused by a typo
    # TODO: be more brave when you're debugging the code
    # (1) run the model here --> (2) get the results here
    return ms.run_lstm_model()  # it might be launching for a long time
    # TODO: if this method returns the wrong dictionary than change the runlstmmodel
    # TODO: install and get into Postman tool --> get into it

# api.add_resource(HelloWorld, '/')  # creating mapping


if __name__ == '__main__':
    #ms.run_lstm_model()
    app.run(debug=True)
