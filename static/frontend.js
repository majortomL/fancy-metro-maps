let map = {}
let zoom = 13

let mapboxToken = "pk.eyJ1IjoibWFqb3J0b21sIiwiYSI6ImNsM24xcXg1NTBhYXMzZW85Yzd6cHBxbnkifQ.r5sK9krUgU_Efqpb1P6i5w"

let points = [
    [1024134.05090108, 6234448.47820225],
    [1024254.41139169, 6235311.97768427]
]

let info = [
    "Stelle",
    "Geroksruhe"
]

let neckarpark = [1024134.05090108, 6234448.47820225]
neckarpark = convertCoordinates(neckarpark)
setupMap(neckarpark, zoom)

pointsConverted = []
points.forEach(
    point => pointsConverted.push(convertCoordinates(point))
)
drawLine(pointsConverted, 'red', info)


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

function drawLine(points, color, info) {
    L.polyline(
        points,
        {
            color: color,
            opacity: 0.5
        }
    ).addTo(map)

    points.forEach(function (point, i) {
        setMarker(point, color, info[i])
    })
}

function setMarker(point, color, info) {
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
        .addTo(map)
        .bindTooltip(info)
        .on('mouseover', function(){
            this.setStyle({
                radius: 7,
                fillOpacity: 1,
                opacity: 1
            })
        })
        .on('mouseout', function (){
            this.setStyle({
                radius: 5,
                fillOpacity: 0.5,
                opacity: 0.7
            })
        })
}

function convertCoordinates(point) {
    let source = new proj4.Proj('EPSG:3857')
    let dest = new proj4.Proj('EPSG:4326');
    let p = new proj4.Point(point[0], point[1]);
    let transformed = proj4.transform(source, dest, p);
    return [transformed.y, transformed.x]
}