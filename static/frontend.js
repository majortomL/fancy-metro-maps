let slides = []
let slidesArray = []
let slidePosition = 0;

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

// arrays for all station markers of map and graph
let allMarkers = []
let switchPointMarkers = []
let graphMarkers = []
let graphLines = []
let mapMarkers = []
let mapLines = []

// layer for the graph which holds the leaflet map
let graphMapLayer = null

let freiburgCenter = [48, 7.846653]
setupScrollEvents()
setupMap(freiburgCenter, zoom)
setupGraph(freiburgCenter, zoom)

// sync map movement and zoom with graph movement and zoom
map.sync(graph)
// sync graph movement and zoom with map movement and zoom
graph.sync(map)

addRadioButtonEvents()

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
    // L.control.bigImage({position: 'topright'}).addTo(graph)
}

function fitMap() {
    map.fitBounds(new L.featureGroup(allMarkers).getBounds())
}

function drawLine(target, points, color, opacity, offset) {
    let line = L.polyline(
        points,
        {
            color: color,
            opacity: opacity,
            interactive: false,
            offset: offset,
            weight: 3
        }
    ).addTo(target)
    return line
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
    mapMarkers.forEach(function (marker) {
        map.removeLayer(marker)
    })

    mapLines.forEach(function (line) {
        map.removeLayer(line)
    })

    mapMarkers = []
    mapLines = []

    let citySelector = document.getElementById("select-city")
    let city = citySelector.options[citySelector.selectedIndex].value

    getJSON(window.location.href + '/data-map/' + city,
        function (err, data) {
            if (err !== null) {
                alert('Something went wrong: ' + err);
            } else {
                mapData = data
                // mark each station
                data.links.forEach(function (link) { // mark each line
                    link.info.line_color.forEach(function (color, i) {
                        let offsetDict = getOffsetDict(link.info.line_label.length)
                        let linkCoordinates = [convertCoordinates([link.source.coord_x, link.source.coord_y]), convertCoordinates([link.target.coord_x, link.target.coord_y])]
                        let flipFactor = getFlipFactor(linkCoordinates)
                        let line = drawLine(
                            map,
                            linkCoordinates,
                            '#' + color,
                            1.0,
                            flipFactor * 4 * offsetDict[i]
                        )
                        mapLines.push(line)
                    })
                })
                data.nodes.forEach(function (node) { // mark each line
                    if (!node.label.includes("Switch Point")) {
                        let marker = drawMarker(map, convertCoordinates(node.pos), 'white', node.label)
                        allMarkers.push(marker)
                        mapMarkers.push(marker)
                    }
                    // if (node.label.includes("Switch Point")) { // TODO: Remove this to disable switch points on map
                    //     let marker = drawMarker(map, convertCoordinates(node.pos), 'white', node.label)
                    //
                    //     switchPointMarkers.push(marker)
                    // }
                })
                fitMap()
            }
        }
    )
}

function drawMetroGraph() { // draws our metro map in the octilinear graph layout
    document.getElementById('calculate-button').disabled = true
    document.getElementById('spinner').style.visibility = 'visible';

    graphMarkers.forEach(function (marker) {
        graph.removeLayer(marker)
    })

    graphLines.forEach(function (line) {
        graph.removeLayer(line)
    })

    allMarkers = []
    graphMarkers = []
    graphLines = []

    drawMetroMap()

    let preCalculated = document.getElementById('realtime-calculation-off').checked
    let citySelector = document.getElementById('select-city')
    let city = citySelector.options[citySelector.selectedIndex].value
    let gridResolution = document.getElementById('grid-size-input').value

    getJSON(window.location.href + '/data-graph/' + city + '/' + preCalculated + '/' + gridResolution,
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
                            graphLines.push(drawLine(
                                graph,
                                linkCoordinates,
                                '#' + color,
                                1.0,
                                flipFactor * 4 * offsetDict[i]
                            ))
                        })
                    }
                })

                data.nodes.forEach(function (node) { // mark each station
                    if (node.isStation && !node.stationInfo.station_label.includes("Switch Point")) {
                        let marker = drawMarker(graph, convertCoordinates(node.pos), 'white', node.stationInfo.station_label)
                        allMarkers.push(marker)
                        graphMarkers.push(marker)
                    }
                    // if (node.isStation && node.stationInfo.station_label.includes("Switch Point")) { // TODO: Remove this to disable switch points on graph
                    //     let marker = drawMarker(graph, convertCoordinates(node.pos), 'white', node.stationInfo.station_label)
                    //     switchPointMarkers.push(marker)
                    // }
                })
            }
            document.getElementById('calculate-button').disabled = false
            document.getElementById('spinner').style.visibility = 'hidden';
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

function setupScrollEvents() {
    slides = document.getElementsByClassName('slide')
    for(i=0; i<slides.length; i++){
        slidesArray.push(slides[i])
    }
    slidesArray.push(document.getElementById('final-slide'))

    document.addEventListener('wheel', function (e) {
        if (e.deltaY > 0) { // user scrolled down
            if (slidesArray[slidePosition + 1] != null) {
                slidePosition++
            }
        } else if (e.deltaY < 0) { // user scrolled up
            if (slidesArray[slidePosition - 1] != null) {
                slidePosition--
            }
        }
        window.scrollTo({top: slidesArray[slidePosition].offsetTop, behavior: 'smooth'})
    })
}

