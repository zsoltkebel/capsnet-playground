import * as d3 from "d3";
import { CapsuleNetwork } from "./capsnet";
import * as tf from "@tensorflow/tfjs";

const NETWORK_HEIGHT = 500;
const NETWORK_WIDTH = 700;

const CAPSULE_HEIGHT = 30;
const CAPSULE_WIDTH = 60;

const RECT_SIZE = 50;

/**
* 
* @param {Capsule[][]} network 
*/
function drawNetwork(network) {
    let width = Math.max(0.9 * window.innerWidth, 600); //TODO calculate from DOM
    const padding = 50;
    
    const svg = d3.select("svg");
    svg.attr("width", width);
    svg.attr("height", 500);
    
    
    // Remove all svg elements.
    // svg.selectAll("*").remove();
    // Remove all div elements.
    d3.select("#network").selectAll("div.capsule").remove();
    d3.select("#network").selectAll("div.plus-minus-neurons").remove();
    
    
    let capsuleCoordinates = {};
    const layerCount = network.length;
    
    let nodeIndexScale = (nodeIndex) => nodeIndex * (CAPSULE_HEIGHT + 25);
    let layerScale = d3.scaleLinear([0, layerCount - 1], [CAPSULE_WIDTH / 2 + padding, width - CAPSULE_WIDTH / 2 - padding]); //TODO set to dynamic width of screen
    
    for (let currentLayer = 0; currentLayer < layerCount; currentLayer++) {
        let cx = layerScale(currentLayer);
        
        for (let currentCapsule = 0; currentCapsule < network[currentLayer].length; currentCapsule++) {
            let capsule = network[currentLayer][currentCapsule];
            
            let cy = nodeIndexScale(currentCapsule) + CAPSULE_HEIGHT / 2;
            
            capsuleCoordinates[capsule.id] = {
                cx: cx,
                cy: cy
            };
            drawNode(cx, cy, capsule.id);
            
            // for (let currentLink = 0; currentLink < capsule.outputLinks.length; currentLink++) {
            //     let iter = Math.min(2, Math.max(0, iteration - currentLayer * 3)); //TODO do not hardcode 3 iteration 
            //     drawLink(
            //         cx + CAPSULE_WIDTH / 2,
            //         cy,
            //         layerScale(currentLayer + 1) - CAPSULE_WIDTH / 2,
            //         nodeIndexScale(currentLink) + CAPSULE_HEIGHT / 2,
            //         capsule.outputLinks[currentLink],
            //         iter
            //     );
            // }
        }
        
        addPlusMinusControl(cx, currentLayer + 1);
    }

    updateLinks(capsuleCoordinates, capsnet.allLinks);
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

/**
 * 
 * @param {{[id: string]: {cx: number, cy: number}}} nodeCoordinates 
 * @param {Link[]} links 
 */
function updateLinks(nodeCoordinates, links) {
    // Create a link generator
    const linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

    const tooltip = d3.select("svg").append("text")
    .attr("font-size", "14px")
    .attr("fill", "black")
    .attr("visibility", "hidden");

    const opacityScale = d3.scaleLinear([0, 1], [0.05, 1.0]);

    const linksD3 = d3.select("svg")
        .select("#links")
        .selectAll("path")
        .data(links)
        .join("path")
        .attr("d", (link) => {
            // Define two points
            const source = {x: nodeCoordinates[link.source.id].cx, y: nodeCoordinates[link.source.id].cy};
            const target = {x: nodeCoordinates[link.destination.id].cx, y: nodeCoordinates[link.destination.id].cy};

            return linkGen({ source, target });
        })
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", 2)
    linksD3
        .transition()
        .attr("opacity", (link, i) => {
            const linkLayer = capsnet.linkLayer(i)
            // console.log(linkLayer)
            let iter = Math.min(2, Math.max(0, iteration - linkLayer * 3)); //TODO do not hardcode 3 iteration 
            return d3.scaleLinear([0, 1 / link.source.outputLinks.length], [0.05, 1.0])(link.couplingCoefficient[iter]);
        });
    
    linksD3
        .on("mousemove", function(event, d) {
            const [x, y] = d3.pointer(event);
            
            tooltip.attr("x", x + 10)  // Offset to avoid cursor overlap
            .attr("y", y)
            .text(`${d.couplingCoefficient[0].toFixed(4)}`) //TODO don't hardcode 0, it should be depending on the iteration slider
            .attr("visibility", "visible");
        })
        .on("mouseleave", function() {
            tooltip.attr("visibility", "hidden");
        })
        

        console.log();
}

/**
* 
* @param {*} source_x 
* @param {*} source_y 
* @param {*} target_x 
* @param {*} target_y 
* @param {Link} link 
*/
function drawLink(source_x, source_y, target_x, target_y, link, iteration) {
    const svg = d3.select("svg");
    
    // Define two points
    const source = {x: source_x, y: source_y};
    const target = {x: target_x, y: target_y};
    
    // Create a link generator
    const linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
    
    const tooltip = svg.append("text")
    .attr("font-size", "14px")
    .attr("fill", "black")
    .attr("visibility", "hidden");
    
    const opacityScale = d3.scaleLinear([0, 1], [0.05, 1.0]);
    const cValue = link.couplingCoefficient[iteration]; //TODO shouldn't be hardcoded (iteration step)
    
    // Append the path
    svg.append("path")
    .classed("link", true)
    .attr("d", linkGen({ source, target }))
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 2)
    .attr("opacity", opacityScale(cValue)) //TODO change based on contribution
    
    svg.append("path")
    .attr("d", linkGen({ source, target }))
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("opacity", 0.0)
    .attr("stroke-width", 8)// thicker line for hover 
    .on("mousemove", function(event) {
        const [x, y] = d3.pointer(event);
        
        tooltip.attr("x", x + 10)  // Offset to avoid cursor overlap
        .attr("y", y)
        .text(`${cValue}`)
        .attr("visibility", "visible");
    })
    .on("mouseleave", function() {
        tooltip.attr("visibility", "hidden");
    });
}

function drawNode(cx, cy, nodeId, isInput, container, node) {
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
    .style("line-height", `${CAPSULE_HEIGHT}px`)
    .style("height", `${CAPSULE_HEIGHT}px`)
    .style("width", `${CAPSULE_WIDTH}px`)
    // .html("c");
    
    
}

function addPlusMinusControl(x, layerIdx) {
    let div = d3.select("#network").append("div")
    .classed("plus-minus-neurons", true)
    .style("left", `${x - 50}px`);
    
    let i = layerIdx - 1;
    let firstRow = div.append("div").attr("class", `ui-numNodes${layerIdx}`);
    firstRow.append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => {
        if (capsuleCounts[i] >= 8) {
            return;
        }
        capsuleCounts[i]++;
        reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("add");
    
    firstRow.append("button")
    .attr("class", "mdl-button mdl-js-button mdl-button--icon")
    .on("click", () => {
        if (capsuleCounts[i] <= 1) {
            return;
        }
        capsuleCounts[i]--;
        reset();
    })
    .append("i")
    .attr("class", "material-icons")
    .text("remove");
    
    let suffix = capsuleCounts[i] > 1 ? "s" : "";
    div.append("div").text(
        capsuleCounts[i] + " capsule" + suffix
    );
    
    div.append("input")
    .attr("id", `${x}`)
    .attr("type", "number")
    .attr("step", 1)
    .attr("min", 1)
    .attr("max", 10)
    .attr("value", capsuleDimensions[i])
    .on("input", (event) => {
        const newValue = event.target.value;
        const validValue = clamp(1, 10);//TODO hardcoded values
        const v = validValue(newValue);
        
        capsuleDimensions[i] = v;
        reset();
    });
}

function clamp(min, max) {
    return (value) => Math.min(max, Math.max(min, value));
}

let capsuleCounts = [3, 2, 5];
let capsuleDimensions = [2, 2, 3];

let capsnet = CapsuleNetwork.random(
    capsuleCounts, // capsule counts per layer
    capsuleDimensions  // capsule dimensions per layer
);

let iteration = 0;

// draw();
// drawNode(20, 20, 0);

// drawNode(20, 50, 0);

window.addEventListener("resize", () => drawNetwork(capsnet.network));

async function reset() {
    iterationControl();

    capsnet = CapsuleNetwork.random(
        capsuleCounts, // capsule counts per layer
        capsuleDimensions  // capsule dimensions per layer
    );

    console.log(capsnet.network);
    capsnet.forward();
    // capsnet.dynamicRouting(3, 0);
    console.log(capsnet);
    
    drawNetwork(capsnet.network);
}

reset();

// const config = [
//     {
//         capsules: tf.tensor([
//             [0.5, 1.0],
//             [1.0, 1.0],
//             [1.0, 2.0]
//         ]),
//         weights: tf.tensor([
            
//         ])
//     },
//     {
//         capsules: tf.tensor(),
//         weights: tf.tensor()
//     }
// ]

async function newNetwork() {
    const capsuleLayers = [
        // Layer 0
        tf.tensor([
            [0.5, 1.0],
            [1.0, 1.0],
            [1.0, 2.0]
        ]),
        // Layer 1
        tf.tensor([
            [0.0, 0.0],
            [0.0, 0.0]
        ])
    ];
    
    const weights = [
        tf.tensor([
            [[[ 0.1171, -0.0389],
            [-0.9342, -0.7069]],
            
            [[-2.0736,  1.2055],
            [ 0.7447,  0.8000]]],
            
            
            [[[ 0.1478,  0.2445],
            [-1.0649, -1.3133]],
            
            [[-0.8738, -1.1456],
            [-0.0239, -0.1380]]],
            
            
            [[[-0.1802, -1.3469],
            [ 1.2548,  0.0767]],
            
            [[ 0.4229, -0.3756],
            [-0.4761,  0.6286]]]
        ])
    ];

    const capsnet = new CapsuleNetwork(capsuleLayers, weights);
    console.log(capsnet.network);
    capsnet.forward();
    // capsnet.dynamicRouting(3, 0);
    console.log(capsnet);
    
    drawNetwork(capsnet.network);
}
// newNetwork();

function iterationControl() {
    const iterations = (capsnet.layerCount - 1) * 3 - 1; //TODO do not hardcode 3 iterations
    iteration = iterations;
    d3.select("#iteration")
        .attr("min", 0)
        .attr("max", iterations)
        .attr("step", 1)
        .property("value", iteration)
        .on("input", (event) => {
            const value = Number(event.target.value);
            iteration = value;
            drawNetwork(capsnet.network);

            d3.select('label[for="iteration"]')
                .html(`${iteration + 1}. iteration`)
        })
    
    d3.select('label[for="iteration"]')
        .html(`${iteration + 1}. iteration`)
}

// Layer controls
d3.select("#add-layer")
    .on("click", () => {
        if (capsuleCounts.length >= 10) {
            return;
        }

        capsuleCounts.push(2);
        capsuleDimensions.push(2);

        reset();
    });

d3.select("#remove-layer")
.on("click", () => {
    if (capsuleCounts.length <= 2) {
        return;
    }

    capsuleCounts.pop();
    capsuleDimensions.pop();

    reset();
});