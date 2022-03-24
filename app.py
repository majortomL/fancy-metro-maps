import json
import flask

app = flask.Flask(__name__)


@app.route('/')
def index():  # put application's code here
    return flask.render_template("index.html")


@app.route('/data')
def load_data():
    array = [(1, 5), ("Haus", "Maus"), [3, 4]]
    return json.dumps(array)


if __name__ == '__main__':
    app.run()
