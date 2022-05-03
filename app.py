# -*- encoding: iso-8859-15 -*-
import json
import flask
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

app = flask.Flask(__name__)
data_path = 'data/'
json_file = 'freiburg.json'

stations = []
edges = []

cost_dictionary = {
    180: 0,
    135: 1,
    90: 1.5,
    45: 2
}

c_180 = 0
c_135 = 1
c_90 = 2
c_45 = 3

c_h = 1
c_H = 0  # c_h - a = 0; a = 1 (corresponds to c_h' in paper)
c_s = 10


class Station:
    SWITCH_POINT_COUNTER = 1

    def __init__(self, featureData):
        self.coord_x = featureData['geometry']['coordinates'][0]
        self.coord_y = featureData['geometry']['coordinates'][1]
        self.degree = featureData['properties']['deg']
        self.degree_in = featureData['properties']['deg_in']
        self.degree_out = featureData['properties']['deg_out']
        self.id = featureData['properties']['id']
        try:
            self.station_label = featureData['properties']['station_label']
        except:
            self.station_label = f'Switch Point{Station.SWITCH_POINT_COUNTER}'
            Station.SWITCH_POINT_COUNTER += 1

    def __str__(self):
        print(self.station_label)
        return self.station_label
        # return f"({self.coord_x}, {self.coord_y})"


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
    metro_map = load_data()
    G = octilinear_graph(0, 0, 5, 5, 1)
    A = auxiliary_graph(G)
    pos = nx.get_node_attributes(A, 'pos')
    A = mark_station(0, 0, A)
    A = mark_station(2, 0, A)
    A = mark_edge(((0, 0), (1, 0)), ((1, 0), (0, 0)), A)
    A = mark_edge(((0, 0), (1, 0)), (0, 0), A)

    color_map_nodes = []
    for node in A.nodes:
        if A.nodes[node]['isStation']:
            color_map_nodes.append('red')
        else:
            color_map_nodes.append('blue')

    color_map_edges = []
    for edge in A.edges:
        if A.edges[edge]['isMarked']:
            color_map_edges.append('red')
        else:
            color_map_edges.append('black')

    nx.draw(A, pos, node_size=8, node_color=color_map_nodes, edge_color=color_map_edges)
    plt.show()
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

    g = nx.Graph()

    for edge in edges:
        station_start = next((x for x in stations if x.id == edge.station_from), None)
        station_end = next((x for x in stations if x.id == edge.station_to), None)
        g.add_node(station_start, label=station_start.station_label)
        g.add_node(station_end, label=station_end.station_label)
        g.add_edge(station_start, station_end)

    return g


def mark_station(x, y, G): # Marks a station at position (x,y)
    for node in G.nodes:
        if G.nodes[node]['pos'] == (x, y):
            G.nodes[node]['isStation'] = True
    return G


def mark_edge(a, b, G): # Marks an edge between (a,b)
    for u, v, d in G.edges(data=True):
        if (u == a) & (v == b) or (u == b) & (v == a):
            d['isMarked'] = True
    return G


def octilinear_graph(x1, y1, x2, y2, node_dist):
    g = nx.Graph()

    for x in range(x1, x2 + node_dist, node_dist):
        for y in range(y1, y2 + node_dist, node_dist):
            g.add_node((x, y), pos=(x, y), isStation=False)
            g.add_edge((x, y), (x + node_dist, y))
            g.add_edge((x, y), (x + node_dist, y + node_dist))
            g.add_edge((x, y), (x, y + node_dist))
            g.add_edge((x, y), (x - node_dist, y + node_dist))

    to_delete = []
    for n in g.nodes:
        if g.degree[n] <= 3:
            to_delete.append(n)

    for n in to_delete:
        g.remove_node(n)

    return g


def auxiliary_graph(G: nx.Graph):
    A = nx.Graph()

    # add
    for node in G.nodes:
        intermediate_nodes = []

        for edge in list(G.edges(node)):
            other_node = edge[0] if node == edge[1] else edge[1]
            intermediate_nodes.append((node, other_node))
            A.add_node((node, other_node),
                       pos=(node[0] + (other_node[0] - node[0]) / 3, node[1] + (other_node[1] - node[1]) / 3),
                       isStation=False)
            A.add_node(node, pos=node, isStation=False)
            A.add_edge((node, other_node), (other_node, node), isMarked=False)
            A.add_edge(node, (node, other_node), isMarked=False)

        # make ring of intermediate nodes fully connected
        for i in range(len(intermediate_nodes)):
            for j in range(i + 1, len(intermediate_nodes)):
                A.add_edge(intermediate_nodes[i], intermediate_nodes[j], isMarked=False)

    return A


def compute_cost(num_nodes, graph):
    cost = 0
    nodes = list(graph.edges())

    for i in range(1, num_nodes - 2):
        cost += cost_bend(edges[i - 1], edges[i + 1])

    cost *= (num_nodes - 1) * c_h
    return cost


def cost_bend(edge_a, edge_b):  # Calculates the cost of a line bend based on the angle
    point_0 = edge_a[0]
    point_1 = edge_a[1]
    point_2 = edge_b[0]
    point_3 = edge_b[1]

    vec_0_1 = np.array[point_1[0] - point_0[0], point_1[1] - point_0[1]]
    vec_2_3 = np.array[point_3[0] - point_2[0], point_3[1] - point_2[1]]

    angle_vecs = math.acos((vec_0_1 * vec_2_3) / np.linalg.norm(vec_0_1) * np.linalg.norm(vec_2_3))
    return cost_dictionary[int(angle_vecs)]


if __name__ == '__main__':
    app.run()
