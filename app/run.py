import json
import re

from itertools import chain

import joblib
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# data for the genre visualization
genre_counts = df.groupby('genre').count()['message']
genre_names = list(genre_counts.index)

# data for the categories visualization
category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)

# data for the word usage visualization
word_counts = pd.Series(list(chain.from_iterable(df.message.apply(tokenize).values))).value_counts()[:10]


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    textposition='auto',
                    text=genre_counts,
                    marker=dict(color='#48b393')
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': dict(showgrid=False, visible=False),
            }
        },
        {
            'data': [
                Pie(
                    labels=category_counts.index,
                    values=category_counts.values,
                    hoverinfo='label+value',
                    textposition='inside',
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                },
                'uniformtext': {
                    'minsize': 12,
                    'mode': 'hide',
                },
                'legend': dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            }
        },
        {
            'data': [
                Bar(
                    x=word_counts.index,
                    y=word_counts.values,
                    textposition='auto',
                    text=word_counts.values,
                    marker=dict(color='#48b393')
                )
            ],

            'layout': {
                'title': 'Distribution of word usage',
                'yaxis': dict(showgrid=False, visible=False),
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
