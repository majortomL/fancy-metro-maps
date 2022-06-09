// TODO: allow user to toggle between real time execution and just loading the finished JSON

let map = {}
let graph = {}
let zoom = 13

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

// the json arrays for the map and the graph
let mapData = null
let graphData = null

// arrays for all station markers of map and graph TODO: clear this on map reload for other city
let allMarkers = []
let switchPointMarkers = []
let switchPointLayer = L.layerGroup()

// layer for the graph which holds the leaflet map
let graphMapLayer = null

let freiburgCenter = [48, 7.846653]
setupMap(freiburgCenter, zoom)
setupGraph(freiburgCenter, zoom)

// sync map movement and zoom with graph movement and zoom
map.sync(graph)
// sync graph movement and zoom with map movement and zoom
graph.sync(map)

drawMetroMap()
drawMetroGraph()
addRadioButtonEvents()
addCheckboxEvents()

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
    }).addTo(map)
}

function setupGraph(point, zoom) {
    graph = L.map('graph', {
            attributionControl: false
        }
    ).setView(point, zoom)

    graphMapLayer = L.tileLayer('https://api.mapbox.com/styles/v1/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}', {
        attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
        maxZoom: 18,
        id: 'mapbox/dark-v10',
        tileSize: 512,
        zoomOffset: -1,
        accessToken: mapboxToken
    })
}

function fitMap() {
    map.fitBounds(new L.featureGroup(allMarkers).getBounds())
}

function drawLine(target, points, color, opacity, offset) {
    L.polyline(
        points,
        {
            color: color,
            opacity: opacity,
            interactive: false,
            offset: offset,
            weight: 3
        }
    ).addTo(target)

}

function drawMarker(target, point, color, info) {
    let marker = L.circleMarker(
        point,
        {
            radius: 7,
            fillColor: color,
            color: 'black',
            fillOpacity: 1.0,
            opacity: 1.0,
            weight: 1,
            name: info
        }
    )
        .addTo(target)
        .bindTooltip(info)
        .on('mouseover', function () {
            getMarkerByName(allMarkers, this.options.name).forEach(function (marker) {
                marker.setStyle({
                    radius: 10
                })
                if (marker != this) {
                    marker.openTooltip()
                }
            })
        })
        .on('mouseout', function () {
            getMarkerByName(allMarkers, this.options.name).forEach(function (marker) {
                marker.setStyle({
                    radius: 7
                })
                if (marker != this) {
                    marker.closeTooltip()
                }
            })
        })
    return marker
}

function getMarkerByName(array, name) {
    return array.filter(function (data) {
        return data.options.name == name
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
                data.links.forEach(function (link) { // mark each line
                    link.info.line_color.forEach(function (color, i) {
                        let offsetDict = getOffsetDict(link.info.line_label.length)
                        let linkCoordinates = [convertCoordinates([link.source.coord_x, link.source.coord_y]), convertCoordinates([link.target.coord_x, link.target.coord_y])]
                        let flipFactor = getFlipFactor(linkCoordinates)
                        drawLine(
                            map,
                            linkCoordinates,
                            '#' + color,
                            1.0,
                            flipFactor * 4 * offsetDict[i]
                        )
                    })
                })
                data.nodes.forEach(function (node) { // mark each line
                    if (!node.label.includes("Switch Point")) {
                        allMarkers.push(drawMarker(map, convertCoordinates(node.pos), 'white', node.label))
                    }
                    if (node.label.includes("Switch Point")) { // TODO: Remove this to disable switch points on map
                        let marker = drawMarker(map, convertCoordinates(node.pos), 'white', node.label)

                        switchPointMarkers.push(marker)
                    }
                })
                fitMap()
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

                data.links.forEach(function (link) { // mark each line
                    if (link.line != null) {
                        link.line.line_color.forEach(function (color, i) {
                            let offsetDict = getOffsetDict(link.line.line_label.length)
                            let linkCoordinates = [convertCoordinates(link.source), convertCoordinates(link.target)]
                            let flipFactor = getFlipFactor(linkCoordinates)
                            drawLine(
                                graph,
                                linkCoordinates,
                                '#' + color,
                                1.0,
                                flipFactor * 4 * offsetDict[i]
                            )
                        })
                    }
                })

                data.nodes.forEach(function (node) { // mark each station
                    if (node.isStation && !node.stationInfo.station_label.includes("Switch Point")) {
                        allMarkers.push(drawMarker(graph, convertCoordinates(node.pos), 'white', node.stationInfo.station_label))
                    }
                    if (node.isStation && node.stationInfo.station_label.includes("Switch Point")) { // TODO: Remove this to disable switch points on graph
                        let marker = drawMarker(graph, convertCoordinates(node.pos), 'white', node.stationInfo.station_label)
                        switchPointMarkers.push(marker)
                    }
                })
            }
        })
}

function addRadioButtonEvents() {
    let radioButtons = document.getElementsByName("map-overlay")
    radioButtons.forEach(function (radioButton) {
        radioButton.addEventListener('change', function () {
            if (this.value == 'on') {
                graph.addLayer(graphMapLayer)
            } else {
                graph.removeLayer(graphMapLayer)
            }
        })
    })
}

function getOffsetDict(numLines) {
    let startVal = -(numLines - 1) / 2
    let offsetDict = {0: startVal}

    for (let i = 1; i < numLines; i++) {
        offsetDict[i] = startVal + i
    }
    return offsetDict
}

function getFlipFactor(points) {
    if ((points[1][0] > points[0][0] && points[1][1] <= points[0][1]) ^ points[1][0] <= points[0][0] && points[1][1] < points[0][1]) {
        return -1
    } else return 1
}



