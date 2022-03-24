import json
import flask
import networkx as nx

app = flask.Flask(__name__)
data_path = 'data/'
json_file = 'freiburg.json'

stations = []
edges = []

class Station:
    def __init__(self, featureData):
        self.coord_x = featureData['geometry']['coordinates'][0]
        self.coord_y = featureData['geometry']['coordinates'][1]
        self.degree = featureData['properties']['deg']
        self.degree_in = featureData['properties']['deg_in']
        self.degree_out = featureData['properties']['deg_out']
        self.id = featureData['properties']['id']

class Edge:
    def __init__(self, featureData, i):
        self.coord_x = featureData['geometry']['coordinates'][0]
        self.coord_y = featureData['geometry']['coordinates'][1]
        self.station_from = featureData['properties']['from']
        self.station_to = featureData['properties']['to']
        self.lines_color = featureData['properties']['lines'][i]['color']
        self.line_label = featureData['properties']['lines'][i]['label']


@app.route('/')
def index():
    load_data()
    return flask.render_template("index.html")


@app.route('/data')
def get_data():
    array = [(1, 5), ("Haus", "Maus"), [3, 4]]
    return json.dumps(array)


def load_data():
    data = json.load(open(data_path + json_file))

    for feature in data['features']:
        type = feature['geometry']['type']

        if type == 'Point':
            stations.append(Station(feature))

        if type == 'LineString':
            for i in range(0, len(feature['properties']['lines'])):
                edges.append(Edge(feature, i))

    G = nx.Graph()

    for edge in edges:
        station_start = next((x for x in stations if x.id == edge.station_from), None)
        station_end = next((x for x in stations if x.id == edge.station_to), None)
        G.add_edge(station_start, station_end)

    print("aasdf")


if __name__ == '__main__':
    app.run()
