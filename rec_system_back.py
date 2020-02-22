from flask import Flask
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
api = Api(app)

CORS(app)


@app.route('/', methods=['GET'])
def get():
    return {'hello': 'world'}


@app.route('/number', methods=['GET'])
def number():
    return {'number': 5}

# api.add_resource(HelloWorld, '/')  # creating mapping


if __name__ == '__main__':
    app.run(debug=True)
