from flask import Flask, render_template, request
from base64 import urlsafe_b64encode, urlsafe_b64decode

import os
import logging

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/notes', methods=['POST'])
def make_notes():
    return request.form

@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
