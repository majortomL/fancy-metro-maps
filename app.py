# -*- encoding: iso-8859-15 -*-
import json
import sys

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
c_m = 0.5

A = {}
Shared_Graph = {}
Shared_Map = {}

radius_node_search = 2

CELL_SIZE = 1
Grid_Resolution = 10


class Station:
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
        # print(self.station_label)
        return self.station_label
        # return f"({self.coord_x}, {self.coord_y})"

    def pos_tuple(self):
        return self.coord_x, self.coord_y

    SWITCH_POINT_COUNTER = 0


class Edge:
    def __init__(self, featureData):
        self.coord_x = featureData['geometry']['coordinates'][0]
        self.coord_y = featureData['geometry']['coordinates'][1]
        self.station_from = featureData['properties']['from']
        self.station_to = featureData['properties']['to']
        self.line_color = [line['color'] for line in featureData['properties']['lines']]
        self.line_label = [line['label'] for line in featureData['properties']['lines']]

    def pos_tuple(self):
        return self.coord_x, self.coord_y

    def station_tuple(self):
        return self.station_from, self.station_to

    def line_count(self):
        return len(self.line_label)

    def __str__(self):
        return f"({str(self.station_from)}, {str(self.station_to)})"


class Encoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__


def get_ldeg(G, v):
    edgeDict = nx.get_edge_attributes(G, 'info')
    adjEdges = list(G.edges(v))
    ldeg = 0

    for e in adjEdges:
        try:
            ldeg += edgeDict[e].line_count()
        except KeyError:
            ldeg += edgeDict[(e[1], e[0])].line_count()
    return ldeg
    # return sum([edgeDict[e].line_count() for e in adjEdges])


@app.route('/')
def index():
    global Shared_Graph
    global Shared_Map

    metro_map = load_data()
    Shared_Map = metro_map
    pos = nx.get_node_attributes(metro_map, 'pos')
    nx.draw(metro_map, pos, node_size=8, connectionstyle='arc3, rad = 0.1', with_labels=True)
    plt.show()

    ordered_input_edges = order_input_edges(metro_map)

    # G = octilinear_graph(-1, -2, 3, 2, CELL_SIZE)

    metro_map_extents = get_map_extents(metro_map)
    larger_extent = max(abs(metro_map_extents[0][1] - metro_map_extents[0][0]), abs(metro_map_extents[1][1] - metro_map_extents[1][0]))
    CELL_SIZE = round(larger_extent / Grid_Resolution)

    G = octilinear_graph(-3, -3, 4, 4, CELL_SIZE)
    G = octilinear_graph(metro_map_extents[0][0], metro_map_extents[1][0], metro_map_extents[0][1], metro_map_extents[1][1], CELL_SIZE)

    color_map_edges = []
    for node in G.nodes:
        if G.nodes[node]['isStation']:
            color_map_edges.append('red')
        else:
            color_map_edges.append('black')
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos, node_size=8, node_color=color_map_edges, with_labels=True)
    plt.show()
    global A
    A = auxiliary_graph(G)
    pos = nx.get_node_attributes(A, 'pos')

    G = route_edges(ordered_input_edges, G, metro_map)
    Shared_Graph = G
    show_octilinear_graph(G, False)

    '''
    color_map_nodes = []
    for node in G.nodes:
        if G.nodes[node]['isStation']:
            color_map_nodes.append('red')
        else:
            color_map_nodes.append('blue')

    color_map_edges = []
    for edge in G.edges:
        if 'line' in G.edges[edge]:
            color_map_edges.append('red')
        else:
            color_map_edges.append('black')

    #nx.draw(A, pos, node_size=8, node_color=color_map_nodes, edge_color=color_map_edges, with_labels=True)
    #nx.draw(A, pos, node_size=8, node_color=color_map_nodes, edge_color=color_map_edges)
    nx.draw(G, pos, node_size=8, node_color=color_map_nodes, edge_color=color_map_edges, with_labels=True)
    plt.show()
    
    '''
    return flask.render_template("index.html")


@app.route('/data')
def get_data():
    graph = nx.node_link_data(Shared_Graph)
    map = nx.node_link_data(Shared_Map)
    return json.dumps(map, indent=4, cls=Encoder)


def load_data():
    data = json.load(open(data_path + json_file))

    for feature in data['features']:
        type = feature['geometry']['type']

        if type == 'Point':
            stations.append(Station(feature))

        if type == 'LineString':
            edges.append(Edge(feature))

    g = nx.Graph()

    for edge in edges:
        station_start = next((x for x in stations if x.id == edge.station_from), None)
        station_end = next((x for x in stations if x.id == edge.station_to), None)
        g.add_node(station_start, label=station_start.station_label, pos=station_start.pos_tuple())
        g.add_node(station_end, label=station_end.station_label, pos=station_end.pos_tuple())
        g.add_edge(station_start, station_end, info=edge)

    return g


def mark_station(x, y, G):  # Marks a station at position (x,y)
    for node in G.nodes:
        if G.nodes[node]['pos'] == (x, y):
            G.nodes[node]['isStation'] = True
    return G


def mark_station(node, G):
    for g_node in G.nodes:
        if G.nodes[g_node] == node:
            G.nodes[g_node]['isStation'] = True
    return G


def mark_edge(a, b, G):  # Marks an edge between (a,b)
    for u, v, d in G.edges(data=True):
        if (u == a) & (v == b) or (u == b) & (v == a):
            d['isMarked'] = True
    return G


def mark_edge_line(edge, G, line):
    for g_edge in G.edges():
        if G.edges[g_edge] == edge:
            G.edges[g_edge]['line'] = line
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
            A.add_edge((node, other_node), (other_node, node), isMarked=False, line="")
            A.add_edge(node, (node, other_node), isMarked=False, line="")

        # make ring of intermediate nodes fully connected
        for i in range(len(intermediate_nodes)):
            for j in range(i + 1, len(intermediate_nodes)):
                A.add_edge(intermediate_nodes[i], intermediate_nodes[j], isMarked=False, line="")

    return A


def show_octilinear_graph(Graph, labels):
    G = Graph
    color_map_nodes = []
    for node in G.nodes:
        if G.nodes[node]['isStation']:
            color_map_nodes.append('red')
        else:
            color_map_nodes.append('blue')

    color_map_edges = []
    for edge in G.edges:
        if 'line' in G.edges[edge]:
            color_map_edges.append('red')
        else:
            color_map_edges.append('black')

    for node in G.nodes:
        G.nodes[node]['pos'] = node

    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(dpi=1200)
    nx.draw(G, pos, node_size=8, node_color=color_map_nodes, edge_color=color_map_edges, with_labels=labels)
    plt.show()


def get_other_node(edge, node):
    return edge[1] if edge[0] == node else edge[0]


def order_input_edges(G):
    edge_ordering = []
    unprocessed_nodes = []
    processed_nodes = []

    for node in list(G.nodes):
        unprocessed_nodes.append((node, get_ldeg(G, node)))

    unprocessed_nodes = sorted(unprocessed_nodes, key=lambda tup: tup[1], reverse=True)
    dangling_nodes = [unprocessed_nodes[0]]

    while dangling_nodes:
        dangling_nodes = sorted(dangling_nodes, key=lambda tup: tup[1], reverse=True)
        g_edges = list(G.edges(dangling_nodes[0]))

        intermediate_list = []
        for edge in g_edges:
            other_node = get_other_node(edge, dangling_nodes[0][0])
            other_node = (other_node, get_ldeg(G, other_node))

            if other_node in unprocessed_nodes:
                intermediate_list.append(edge)

        intermediate_list_sorted = sorted(intermediate_list, key=lambda e: get_ldeg(G, get_other_node(e, dangling_nodes[0][0])), reverse=True)
        for e in intermediate_list_sorted:
            other_node = get_other_node(e, dangling_nodes[0][0])
            other_node = (other_node, get_ldeg(G, other_node))
            if other_node not in dangling_nodes:
                dangling_nodes.append(other_node)

        edge_ordering.extend(intermediate_list_sorted)

        if dangling_nodes[0] in unprocessed_nodes:
            unprocessed_nodes.remove(dangling_nodes[0])
        processed_nodes.append(dangling_nodes.pop(0))

    return edge_ordering


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


def route_edges(edges, G, metro_map):
    show_octilinear_graph(G, False)
    grid_node_dict = {}

    # iterate through edges of input graph
    for edge in edges:
        node_0 = edge[0]
        node_1 = edge[1]

        candidate_nodes_0 = []  # nodes in octilinear graph that are near to node_0 (of input graph)
        candidate_nodes_1 = []  # nodes in octilinear graph that are near to node_1 (of input graph)

        # if input nodes came up in previous iterations their position is already fixed
        node_0_free = True
        node_1_free = True
        if node_0 in grid_node_dict.keys():
            candidate_nodes_0.append(grid_node_dict[node_0])
            node_0_free = False
        if node_1 in grid_node_dict.keys():
            candidate_nodes_1.append(grid_node_dict[node_1])
            node_1_free = False

        # for free input nodes add all octilinear graph nodes within certain radius to the
        for node in list(G.nodes):
            if G.nodes[node]['isStation']:
                continue
            if node_0_free and pow(node[0] - node_0.coord_x, 2) + pow(node[1] - node_0.coord_y, 2) < pow(radius_node_search, 2):  # check if octilinear node is within radius around input node
                candidate_nodes_0.append(node)

            if node_1_free and pow(node[0] - node_1.coord_x, 2) + pow(node[1] - node_1.coord_y, 2) < pow(radius_node_search, 2):  # same here
                candidate_nodes_1.append(node)

        if node_0_free and node_1_free:
            # build local Voronoi diagram
            union_candidate_nodes = list(set(candidate_nodes_0 + candidate_nodes_1))
            candidate_nodes_0 = []
            candidate_nodes_1 = []

            for node in union_candidate_nodes:
                closer_node = get_closest_node(node, node_0, node_1)
                if closer_node == node_0:
                    candidate_nodes_0.append(node)
                else:
                    candidate_nodes_1.append(node)
        elif node_0_free and not node_1_free:
            if candidate_nodes_1[0] in candidate_nodes_0:
                candidate_nodes_0.remove(candidate_nodes_1[0])
        elif node_1_free and not node_0_free:
            if candidate_nodes_0[0] in candidate_nodes_1:
                candidate_nodes_1.remove(candidate_nodes_0[0])

        A = fix_weights(G, node_0, candidate_nodes_0, node_1, candidate_nodes_1)

        # find shortest set-set path using dijkstra
        shortest_path_cost = int(sys.maxsize)
        shortest_path = []  # list of edges in octilinear graph
        for node in candidate_nodes_0:
            # find shortest node-set path using dijkstra
            path, path_cost = get_shortest_dijkstra_path_to_set(node, candidate_nodes_1, A, G)
            if path_cost < shortest_path_cost:
                shortest_path = path

        for path_edge in shortest_path:
            G.edges[path_edge]['line'] = metro_map.edges()[edge]['info'].line_label

        final_node0 = shortest_path[0][0]
        final_node1 = shortest_path[-1][1]

        G.nodes[final_node0]['isStation'] = True
        G.nodes[final_node1]['isStation'] = True
        grid_node_dict[node_0] = final_node0
        grid_node_dict[node_1] = final_node1
        show_octilinear_graph(G, False)

    return G  # unsure if I modify per reference or need to return G ... just to be sure I return it


def get_shortest_dijkstra_path_to_set(start_node, target_nodes, A, G):
    # calculate cheapest path from start_node to all nodes (in the auxiliary graph)
    paths = nx.shortest_path_length(A, source=start_node, weight="cost", method="dijkstra")

    # determine the target node with the cheapest path
    cheapest_target = None
    cheapest_path_cost = int(sys.maxsize)
    for target_node in target_nodes:

        path_cost = paths[target_node]
        if path_cost < cheapest_path_cost:
            cheapest_path_cost = path_cost
            cheapest_target = target_node

    # get path to the cheapest target node
    cheapest_path = nx.shortest_path(A, source=start_node, target=cheapest_target, weight="cost", method="dijkstra")

    # convert the cheapest path in the auxiliary graph to a list of nodes of the octilinear graph
    # non-sink-nodes are structured like follows (sink-node, other-sink-node)
    # to convert we check if the sink-node changes and if it does we add it to the path
    octilinear_path_nodes = []
    previous_sink_node = None
    for i in range(1, len(cheapest_path) - 1):
        current_sink_node = cheapest_path[i][0]

        if type(current_sink_node) is tuple:
            if current_sink_node != previous_sink_node:
                octilinear_path_nodes.append(current_sink_node)
            previous_sink_node = current_sink_node

    # convert the node list into an edge list
    octilinear_path = []  # edge list
    for i in range(0, len(octilinear_path_nodes) - 1):
        octilinear_path.append((octilinear_path_nodes[i], octilinear_path_nodes[i + 1]))

    return octilinear_path, cheapest_path_cost


def fix_weights(G, iS, S, iT, T):
    A = auxiliary_graph(G)

    for edge in A.edges:
        n1 = edge[0]
        n2 = edge[1]

        if n1 in S or n2 in S:
            # edge adjacent to set S, set off cost by distance to input node iS
            n = n1 if n1 in S else n2
            A[n1][n2]["cost"] = get_auxiliary_edge_cost(edge) * get_dist_ol_grid_node_to_input_node(n, iS) / CELL_SIZE
        elif n1 in T or n2 in T:
            # edge adjacent to set T, set off cost by distance to input node iT
            n = n1 if n1 in T else n2
            A[n1][n2]["cost"] = get_auxiliary_edge_cost(edge) * get_dist_ol_grid_node_to_input_node(n, iT) / CELL_SIZE
        else:
            # edge not adjacent to any set, use default cost
            A[n1][n2]["cost"] = get_auxiliary_edge_cost(edge)

    return A


def get_dist_ol_grid_node_to_input_node(ol_grid_node, input_node):
    p1 = (input_node.coord_x, input_node.coord_y)
    p2 = ol_grid_node
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)


def get_auxiliary_edge_cost(edge):
    n1 = edge[0]
    n2 = edge[1]

    # find out which kind edge we are dealing with and assign cost based on that
    # possible edge types: sink edge, bend edge, connecting edge

    n1_is_sink = not type(n1[0]) is tuple  # one node of a sink edge is a sink node
    n2_is_sink = not type(n2[0]) is tuple  # while other nodes are tuples of tuples sink nodes are only a tuple

    if n1_is_sink or n2_is_sink:
        # this is a sink edge
        return c_h + c_m  # NOTE unsure maybe c_s

    # bend edges connect two nodes connected to the same sink node
    if n1[0] == n2[0]:  # since nodes adjacent to a sink node are identified with the tuple (sink node, other node) we check if the nodes share the same sink node
        # this is a bend edge
        return cost_dictionary[get_auxiliary_bend_angle(edge)]

    # this is a connecting edge
    return c_h + c_m


def get_auxiliary_bend_angle(edge):
    point_0 = edge[0][0]
    point_1 = edge[0][1]
    point_2 = edge[1][0]
    point_3 = edge[1][1]

    vec_0_1 = np.array((point_1[0] - point_0[0], point_1[1] - point_0[1]))
    vec_2_3 = np.array((point_3[0] - point_2[0], point_3[1] - point_2[1]))

    # from https://www.atqed.com/numpy-vector-angle
    a = vec_0_1
    b = vec_2_3

    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    return int(round(deg))


def get_closest_node(node, check_node_1, check_node_2):  # node: octilinear grid graph node, check_node_1, 2: ordered nodes of input graph
    if pow(node[0] - check_node_1.coord_x, 2) + pow(node[1] - check_node_1.coord_y, 2) <= pow(node[0] - check_node_2.coord_x, 2) + pow(node[1] - check_node_2.coord_y, 2):
        return check_node_1
    else:
        return check_node_2


def get_map_extents(map_graph):
    positions = list(nx.get_node_attributes(map_graph, 'pos').values())

    x_min = positions[0][1]
    x_max = positions[0][1]
    y_min = positions[0][0]
    y_max = positions[0][0]

    for position in positions:
        if position[1] < x_min: x_min = position[1]
        if position[1] > x_max: x_max = position[1]
        if position[0] < y_min: y_min = position[0]
        if position[0] > y_max: y_max = position[0]

    return [[round(y_min), round(y_max)], [round(x_min), round(x_max)]]


if __name__ == '__main__':
    app.run()
