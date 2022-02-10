
from flask import Flask
from version import VERSION

app = Flask(__name__)

from solvepnp import solvePnpFromHttpGet, solvePnpFromHttpPost

def appendVersionHttpHeader(response):
    response.headers['X-SOLVEPNP-VERSION'] = VERSION

@app.route('/')
def index():
    response = app.make_response('Service SolvePNP ' + VERSION)
    appendVersionHttpHeader(response)
    return response

@app.route('/calculate', methods=['GET'])
def calculateHttpGet():
    response = app.make_response(solvePnpFromHttpGet())
    appendVersionHttpHeader(response)
    return response

@app.route('/calculate', methods=['POST'])
def calculateHttpPost():
    response = app.make_response(solvePnpFromHttpPost())
    appendVersionHttpHeader(response)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
