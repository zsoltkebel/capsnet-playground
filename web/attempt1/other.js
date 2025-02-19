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
        // this.values = Array.from({ length: dimension }, () => Math.random());
        this.values = tf.randomNormal([dimension]);
        this.outputLinks = [];
        this.inputLinks = [];
    }

    get dimension() {
        return this.values.size;
    }
}

class Link {
    /**
     * 
     * @param {Capsule} source 
     * @param {Capsule} destination 
     */
    constructor(source, destination) {
        this.source = source;
        this.destination = destination;
        this.weight = tf.randomNormal([destination.dimension, source.dimension]); //W_ij is a matrix of size (dest dim x source dim)
        this.c = []; // c_ij matrix value for this link (this is an array to show value at each iteration of the dynamic routing)
    }

    get u_hat() {
        return tf.matMul(this.weight, tf.expandDims(this.source.values, 1)).squeeze();
    }
}

// Squashing function for Capsule Networks
function squash(s) {
    // Step 1: Compute the norm (Euclidean norm) of the vector along the last axis (axis=-1)
    const norm = tf.norm(s, 'euclidean', -1); // This returns a tensor of shape [batchSize] or a scalar if s is 1D
    
    // Step 2: Square the norm
    const normSquared = tf.square(norm);
    
    // Step 3: Compute the squashing factor: ||s_j||^2 / (1 + ||s_j||^2)
    const squashFactor = tf.div(normSquared, tf.add(normSquared, tf.scalar(1.0)));
    
    // Step 4: Normalize the vector s_j by dividing by its norm
    // If s is a batch of vectors, norm has shape [batchSize], so we need to expand it to shape [batchSize, 1]
    const normalizedS = tf.div(s, norm.expandDims(-1)); // Ensure broadcasting works properly
    
    // Step 5: Return the final squashed vector: squashFactor * normalizedS
    return tf.mul(squashFactor.expandDims(-1), normalizedS);
}

class CapsuleNetwork {
    constructor(capsuleCounts, capsuleDimensions) {
        this.network = buildNetwork(capsuleCounts, capsuleDimensions);
        this.c_ij = {}; //tensor of c_ij matrices where the first dimension will be layer index
    }
    
    forward() {
        for (let layer = 0; layer < this.network.length - 1; layer++) {
            this.dynamicRouting(3, layer);
        }
    }

    dynamicRouting(r, l) {
        const i_count = this.network[l].length;
        const j_count = this.network[l + 1].length;

        let u_hat_2 = tf.stack(this.network[l].map(capsule => {
            return tf.stack(capsule.outputLinks.map(link => link.u_hat));
        }));

        console.log("u_hat_2:");
        u_hat_2.print();

        let b_ij = tf.zeros([i_count, j_count]);
        console.log("b_ij:");
        b_ij.print();

        let c_ij
        let v_j;
        for (let iteration = 0; iteration < r; iteration++) {
            console.log(`Iteration ${iteration + 1}.`);

            c_ij = tf.softmax(b_ij);
            console.log("c_ij:");
            c_ij.print();
            
            // save c_ij values into model for visualisation
            let c_ij_array = c_ij.arraySync();
            for (let ci = 0; ci < c_ij_array.length; ci++) {
                for (let li = 0; li < c_ij_array[ci].length; li++) {
                    this.network[l][ci].outputLinks[li].c.push(c_ij_array[ci][li]);
                }
            }

            //TODO loop approach
            // let s_j = [];
            // for (let j = 0; j < j_count; j++) {
            //     let sum = tf.zeros([this.network[l + 1][j].dimension]);
            //     for (let i = 0; i < i_count; i++) {
            //         let cValue =c_ij.arraySync()[i][j];
            //         // console.log(`c_${i}_${j}: `, cValue);

            //         console.log("u_hat:");
            //         this.network[l][i].outputLinks[j].u_hat.print();

            //         let p = tf.mul(cValue, this.network[l][i].outputLinks[j].u_hat);
            //         sum = tf.add(sum, p);
            //     }
            //     s_j.push(sum);
            //     // console.log("sum:");
            //     // sum.print();
            // }
            // s_j = tf.stack(s_j);
            // console.log("s_j:");
            // s_j.print();
            //TODO tensorflow approach
            let s_j = tf.sum(tf.mul(c_ij.expandDims(2), u_hat_2), 0);
            // s_j = s_j.expandDims(2);
            console.log("s_j:");
            s_j.print();
            

            v_j = squash(s_j);
            console.log("v_j:");
            v_j.print();

            b_ij = tf.add(b_ij, tf.sum(tf.mul(u_hat_2, v_j), 2));
        }
        this.c_ij[l] = c_ij;

        

        return v_j;
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
                    nodeIndexScale(currentLink) + CAPSULE_HEIGHT / 2,
                    capsule.outputLinks[currentLink]
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
    const cValue = link.c[2]; //TODO shouldn't be hardcoded (iteration step)

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

const capsnet = new CapsuleNetwork(
    [3, 2, 5], // capsule counts per layer
    [2, 2, 3]  // capsule dimensions per layer
);
console.log(capsnet.network);
capsnet.forward();
// capsnet.dynamicRouting(3, 0);
console.log(capsnet);

drawNetwork(capsnet.network);