import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

const NETWORK_HEIGHT = 500;
const NETWORK_WIDTH = 700;

const CAPSULE_HEIGHT = 30;
const CAPSULE_WIDTH = 50;

const RECT_SIZE = 50;

const capsules = [0, 2, 3];

let network = d3.select("#network")


class Capsule {
    constructor(id, dimension) {
        this.id = id;
        this.values = Array.from({ length: dimension }, () => Math.random());
        this.outputLinks = [];
        this.inputLinks = [];
    }
}

class Link {
    constructor(source, destination) {
        this.source = source;
        this.destination = destination;
    }
}

/**
 * 
 * @param {number[]} capsuleCounts 
 * @param {number[]} capsuleDimensions 
 * @returns 
 */
function buildNetwork(capsuleCounts, capsuleDimensions) {

    let network = []

    for (let i = 0; i < capsuleCounts.length; i++) {
        let layer = [];
        network.push(layer);

        for (let j = 0; j < capsuleCounts[i]; j++) {
            const id = `capsule-${i}-${j}`;
            let capsule = new Capsule(id, capsuleDimensions[i]);
            layer.push(capsule);

            if (i >= 1) {
                // add links to next layer
                for (let k = 0; k < capsuleCounts[i - 1]; k++) {
                    let previousCapsule = network[i - 1][k];
                    let link = new Link(previousCapsule, capsule);
                    previousCapsule.outputLinks.push(link);
                    capsule.inputLinks.push(link);
                }
            }
        }
    }

    return network
}





// network.selectAll("div")
//     .data(capsules)
//     .join("div")
//     .attr("id", "capsule")
//     .style({
//         top: 10,
//     })
//     .html("hello");

/**
 * 
 * @param {Capsule[][]} network 
 */
function drawNetwork(network) {
    let width = 800; //TODO calculate from DOM
    const padding = 50;

    const svg = d3.select("svg");
    svg.attr("width", width);
    svg.attr("height", 500);

    const layerCount = network.length;

    let nodeIndexScale = (nodeIndex) => nodeIndex * (CAPSULE_HEIGHT + 25);
    let layerScale = d3.scaleLinear([0, layerCount - 1], [CAPSULE_WIDTH / 2 + padding, width - CAPSULE_WIDTH / 2 - padding]); //TODO set to dynamic width of screen

    for (let currentLayer = 0; currentLayer < layerCount; currentLayer++) {
        let cx = layerScale(currentLayer);

        for (let currentCapsule = 0; currentCapsule < network[currentLayer].length; currentCapsule++) {
            let capsule = network[currentLayer][currentCapsule];

            let cy = nodeIndexScale(currentCapsule) + CAPSULE_HEIGHT / 2;

            drawNode(cx, cy, capsule.id);

            for (let currentLink = 0; currentLink < capsule.outputLinks.length; currentLink++) {
                drawLink(
                    cx + CAPSULE_WIDTH / 2,
                    cy,
                    layerScale(currentLayer + 1) - CAPSULE_WIDTH / 2,
                    nodeIndexScale(currentLink) + CAPSULE_HEIGHT / 2
                );
            }
        }
    }
}

function draw() {
    let width = 500; //TODO calculate from DOM

    const svg = d3.select("svg");
    svg.attr("width", width);


    // Map of all node coordinates.
    // let container = svg.append("g")
    //     .classed("core", true)
    //     .attr("transform", `translate(${padding},${padding})`);
    // Draw the network layer by layer.
    let numLayers = 2;
    let featureWidth = 118;
    let layerScale = d3.scaleOrdinal()
        .domain(d3.range(1, numLayers - 1))
        .range([featureWidth, width - RECT_SIZE], 0.7);
    let nodeIndexScale = (nodeIndex) => nodeIndex * (CAPSULE_HEIGHT + 25);

    // Draw the input layer separately.
    let cx = CAPSULE_WIDTH / 2 + 50;
    let nodeIds = capsules;
    let maxY = nodeIndexScale(nodeIds.length);
    nodeIds.forEach((nodeId, i) => {
        let cy = nodeIndexScale(i) + CAPSULE_HEIGHT / 2;
        // node2coord[nodeId] = {cx, cy};
        // drawNode(cx, cy, nodeId, true, container);
        drawNode(cx, cy, nodeId);
        drawLink(cx + CAPSULE_WIDTH / 2, cy, 400, 50);
    });

    nodeIds.forEach((nodeId, i) => {
        let cy = nodeIndexScale(i) + CAPSULE_HEIGHT / 2;
        // node2coord[nodeId] = {cx, cy};
        // drawNode(cx, cy, nodeId, true, container);
        drawNode(cx, cy, nodeId);
        drawLink(cx + CAPSULE_WIDTH / 2, cy, 400, 50);
    });
}

function getCoordinatesOfCapsule(layer, index) {

}

function drawLink(source_x, source_y, target_x, target_y) {
    const svg = d3.select("svg");
    
    // Define two points
    const source = {x: source_x, y: source_y};
    const target = {x: target_x, y: target_y};

    // Create a link generator
    const link = d3.linkHorizontal()
        .x(d => d.x)
        .y(d => d.y);

    // Append the path
    svg.append("path")
        .attr("d", link({ source, target }))
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", 2)
        .attr("opacity", 0.5); //TODO change based on contribution
}

function drawNode(cx, cy, nodeId, isInput,
    container, node) {
    let x = cx - CAPSULE_WIDTH / 2;
    let y = cy - CAPSULE_HEIGHT / 2;

    // let nodeGroup = container.append("g")
    //     .attr({
    //     "class": "node",
    //     "id": `node${nodeId}`,
    //     "transform": `translate(${x},${y})`
    //     });

    // // Draw the main rectangle.
    // nodeGroup.append("rect")
    //     .attr({
    //     x: 0,
    //     y: 0,
    //     width: RECT_SIZE,
    //     height: RECT_SIZE,
    //     });

    // Draw the node's canvas.
    let div = d3.select("#network").insert("div", ":first-child")
        .attr("id", `capsule-${nodeId}`)
        .attr("class", "capsule")
        .style("position", "absolute")
        .style("left", `${x - 2}px`)
        .style("top", `${y - 2}px`)
        .style("height", `${CAPSULE_HEIGHT}px`)
        .style("width", `${CAPSULE_WIDTH}px`)
        .html("c");

   
}

// draw();
// drawNode(20, 20, 0);

// drawNode(20, 50, 0);

const capsnet = buildNetwork([3, 2], [2, 2]);
console.log(capsnet);

drawNetwork(capsnet);