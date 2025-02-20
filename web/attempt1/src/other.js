import * as d3 from "d3";
import { CapsuleNetwork } from "./capsnet"

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
    let width = 800; //TODO calculate from DOM
    const padding = 50;
    
    const svg = d3.select("svg");
    svg.attr("width", width);
    svg.attr("height", 500);
    
    
    // Remove all svg elements.
    svg.selectAll("*").remove();
    // Remove all div elements.
    d3.select("#network").selectAll("div.capsule").remove();
    d3.select("#network").selectAll("div.plus-minus-neurons").remove();
    
    
    
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
                    nodeIndexScale(currentLink) + CAPSULE_HEIGHT / 2,
                    capsule.outputLinks[currentLink]
                );
            }
        }
        
        addPlusMinusControl(cx, currentLayer + 1);
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

/**
* 
* @param {*} source_x 
* @param {*} source_y 
* @param {*} target_x 
* @param {*} target_y 
* @param {Link} link 
*/
function drawLink(source_x, source_y, target_x, target_y, link) {
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
    const cValue = link.couplingCoefficient[2]; //TODO shouldn't be hardcoded (iteration step)
    
    // Append the path
    svg.append("path")
    .attr("d", linkGen({ source, target }))
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 2)
    .attr("opacity", opacityScale(cValue)) //TODO change based on contribution
    
    .on("mousemove", function(event) {
        const [x, y] = d3.pointer(event);
        
        tooltip.attr("x", x + 10)  // Offset to avoid cursor overlap
        .attr("y", y - 10)
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
    .html("c");
    
    
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

let capsuleCounts = [3, 2];
let capsuleDimensions = [2, 2];

// draw();
// drawNode(20, 20, 0);

// drawNode(20, 50, 0);

async function reset() {
    const capsnet = new CapsuleNetwork(
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