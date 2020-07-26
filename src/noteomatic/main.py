from flask import Flask, \
                render_template, \
                request, \
                redirect, \
                url_for, \
                Response

import os
import io
import logging
import nltk

import noteomatic.backend.utils.models as models
import noteomatic.backend.utils.text as text
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
STOPWORDS = nltk.corpus.stopwords.words('english')
TEXTMAXLENGTH = 10000

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/notes', methods=['POST'])
def make_notes():
    paragraphs = request.form['analysis']
    if len(paragraphs) > TEXTMAXLENGTH:
        # Generate error
        return redirect(url_for('/'))
    cleaned = text.clean_text(paragraphs, STOPWORDS)
    sent_tok = nltk.sent_tokenize(paragraphs)
    model, db, clusters, word_clusters, n_clusters, n_noise = models.word2vec_model(list(set(cleaned)), min_count=1, window=5, verbose=True)
    notes = text.create_notes(paragraphs, word_clusters)
    percentage = round(100 * (len(notes) / len(paragraphs)), 3)
    #generate_wordcloud(cleaned, STOPWORDS)
    return render_template("results.html", percentage=percentage, notes=notes)


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]
    axis.plot(xs, ys)
    return fig

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
