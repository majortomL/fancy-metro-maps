let map = {}
let graph = {}
let zoom = 13
let currentPosition;
let currentZoom;

let getJSON = function (url, callback) {
    let xhr = new XMLHttpRequest();
    xhr.open('GET', url, true);
    xhr.responseType = 'json';
    xhr.onload = function () {
        var status = xhr.status;
        if (status === 200) {
            callback(null, xhr.response);
        } else {
            callback(status, xhr.response);
        }
    };
    xhr.send();
};

let mapboxToken = "pk.eyJ1IjoibWFqb3J0b21sIiwiYSI6ImNsM24xcXg1NTBhYXMzZW85Yzd6cHBxbnkifQ.r5sK9krUgU_Efqpb1P6i5w"

let points = [
    [1024134.05090108, 6234448.47820225],
    [1024254.41139169, 6235311.97768427]
]

let info = [
    "Stelle",
    "Geroksruhe"
]

// the json arrays for the map and the graph
let mapData = null
let graphData = null

let neckarpark = [1024134.05090108, 6234448.47820225]
neckarpark = convertCoordinates(neckarpark)

let freiburgCenter = [48, 7.846653]
setupMap(freiburgCenter, zoom)
setupGraph(freiburgCenter, zoom)

// sync map movement and zoom with graph movement and zoom
map.sync(graph)
// sync graph movement and zoom with map movement and zoom
graph.sync(map)

drawMetroMap()

function setupMap(point, zoom) {
    map = L.map('map', {
            attributionControl: false
        }
    ).setView(point, zoom)
    L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}', {
        attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
        maxZoom: 18,
        id: 'mapbox/dark-v10',
        tileSize: 512,
        zoomOffset: -1,
        accessToken: mapboxToken
    }).addTo(map);
}

function setupGraph(point, zoom) {
    graph = L.map('graph', {
            attributionControl: false
        }
    ).setView(point, zoom)
}

function drawLine(target, points, color) {
    L.polyline(
        points,
        {
            color: color,
            opacity: 0.5,
            interactive: false
        }
    ).addTo(target)

}

function drawMarker(target, point, color, info) {
    let marker = L.circleMarker(
        point,
        {
            radius: 5,
            fillColor: color,
            color: color,
            fillOpacity: 0.5,
            opacity: 0.7,
            weight: 1,
        }
    )
        .addTo(target)
        .bindTooltip(info)
        .on('mouseover', function () {
            this.setStyle({
                radius: 7,
                fillOpacity: 1,
                opacity: 1
            })
        })
        .on('mouseout', function () {
            this.setStyle({
                radius: 5,
                fillOpacity: 0.5,
                opacity: 0.7
            })
        })
}

function convertCoordinates(point) {
    let source = new proj4.Proj('EPSG:3857')
    let dest = new proj4.Proj('EPSG:4326')
    let p = new proj4.toPoint([point[0], point[1]])
    let transformed = proj4.transform(source, dest, p)
    return [transformed.y, transformed.x]
}

function drawMetroMap() { // draws our metro map with real station coordinates
    getJSON(window.location.href + '/data-map',
        function (err, data) {
            if (err !== null) {
                alert('Something went wrong: ' + err);
            } else {
                mapData = data
                // mark each station
                let stationCoordinates = []
                data.nodes.forEach(function (node) {
                    if (!node.label.includes("Switch Point")) {
                        drawMarker(map, convertCoordinates(node.pos), 'white', node.label)
                        drawMarker(graph, convertCoordinates(node.pos), 'white', node.label)
                    }
                })
                // mark each line
                data.links.forEach(function (link) {
                    link.info.line_color.forEach(function (color) {
                        drawLine(map, [convertCoordinates([link.source.coord_x, link.source.coord_y]), convertCoordinates([link.target.coord_x, link.target.coord_y])], '#' + color)
                        drawLine(graph, [convertCoordinates([link.source.coord_x, link.source.coord_y]), convertCoordinates([link.target.coord_x, link.target.coord_y])], '#' + color) // TODO: delete this when done building the graph
                    })
                })
            }
        }
    )
}

function drawMetroGraph() { // draws our metro map in the octilinear graph layout
    getJSON(window.location.href + '/data-graph',
        function (err, data) {
            if (err !== null) {
                alert('Something went wrong: ' + err);
            } else {
                graphData = data
                // mark each station
                let stationCoordinates = []
                data.nodes.forEach(function(node){
                    if(node.isStation){
                        drawMarker(graph, convertCoordinates(node.pos), 'black', 'graph point')
                    }
                })
                data.links.forEach(function (link){
                    drawLine(graph, [convertCoordinates(link.source), convertCoordinates(link.target)], 'black')
                })
            }
        })
}



