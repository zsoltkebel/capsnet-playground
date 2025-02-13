import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";

let out_capsule_count = 2;
let out_capsule_dimension = 2;

/**
 * 
 * @param {*} tensor 
 * @param {d3.Selection} layerContainer 
 */
function addCapsules(tensor, layerContainer) {
    // Bind rows (each row is a <g> element)
    const rows = layerContainer.selectAll("div")
        .data(tensor)
        .join("div")
        .classed("capsule", true)
        .text(d => d.map(num => Number(num).toFixed(2)).join(", "));
}

let t = [
    [1, 2, 3, 5],
    [1, 2, 1, 4]
]

let te = tf.tensor([
    [1, 2, 3, 5],
    [1, 2, 1, 4]
])

const layer = d3.select(".layer");

// addCapsules(te.arraySync(), layer)



// stuff about routing
function routing(in_capsules, out_capsule_count, out_capsule_dimension, r=3) {
    let in_capsule_count = in_capsules.shape[0];
    let in_capsule_dimension = in_capsules.shape[1];

    console.log("in caps: " + in_capsule_count + " in caps dims: " + in_capsule_dimension);
    const W = tf.randomNormal([1, in_capsule_count, out_capsule_count, out_capsule_dimension, in_capsule_dimension]);

    // const W = tf.tensor([[[[[-0.2720, -0.3906],
    //      [ 0.6187,  0.4854]],

    //     [[ 1.3667, -0.2222],
    //      [ 0.8012,  2.1180]]],


    //    [[[ 0.0833, -0.2355],
    //      [ 1.2901, -0.9415]],

    //     [[ 0.6626,  0.7623],
    //      [ 0.7971, -1.1760]]],


    //    [[[-1.6190, -0.8066],
    //      [ 1.1043, -0.8219]],

    //     [[ 0.5557, -0.3481],
    //      [ 0.0916,  1.0921]]]]])

    console.log("W");
    W.print();
    
    let b_ij = tf.zeros([in_capsule_count, out_capsule_count]);
    console.log("b_ij");
    b_ij.print();

    let u = in_capsules
      .expandDims(1)
      .broadcastTo([in_capsule_count, out_capsule_count, in_capsule_dimension])
      .expandDims(3);
    console.log("u");
    u.print();
    
    console.log("W shape: " + W.shape);
    console.log("u shape: " + u.shape);

    let u_hat = tf.matMul(W, u);
    // u_hat.squeeze(3);
    console.log("u_hat:");
    u_hat.print();
    u_hat = u_hat.reshape([in_capsule_count, out_capsule_count, in_capsule_dimension]);
    console.log("u_hat:");
    u_hat.print();

    for (let iteration = 0; iteration < r; iteration++) {
      console.log(`iteration ${iteration} ==========`);

      let c_ij = tf.softmax(b_ij);
      c_ij = c_ij.expandDims(2); // expand last dimension
      console.log("c_ij:");
      c_ij.print();

      let s_j = tf.sum(tf.mul(c_ij, u_hat), 0);
      s_j = s_j.expandDims(2);
      console.log("s_j:");
      s_j.print();

      let  v_j = squash(s_j);
      console.log("v_j:" + typeof(v_j));
      v_j.print();
      console.log(v_j.shape)

      let v_j_transformed = v_j.squeeze(2).expandDims(0).broadcastTo([in_capsule_count, out_capsule_count, out_capsule_dimension]);
      console.log("v_j transformed");
      v_j_transformed.print();

      b_ij = tf.add(b_ij, tf.sum(tf.mul(u_hat, v_j_transformed), 2));

      if (iteration == r - 1) {
        return v_j;
      }
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

let u = tf.tensor([
[1.0, 1.0],
[2.0, 0.5],
[1.2, 2.0],
[3, 4]
])

let v = routing(u, out_capsule_count, out_capsule_dimension);

addCapsules(u.arraySync(), layer);
const layer1 = d3.select("#layer1");
addCapsules(v.arraySync(), layer1);

d3.select("#add-layers")
.on("click", () => {
    console.log("clicked");

    const newCapsule = tf.tensor([
        Array.from({ length: u.shape[1] }, () => Math.floor(Math.random() * 5) + 1)
    ])
    u = tf.concat([u, newCapsule]);
    
    drawNetwork();
})

d3.select("#remove-layers")
.on("click", () => {
    u = removeCapsule(u);
    drawNetwork();
})

// output layer buttons
d3.select("#add-output-layer")
.on("click", () => {
    out_capsule_count++;
    drawNetwork();
})

d3.select("#remove-output-layer")
.on("click", () => {
    out_capsule_count--;
    drawNetwork();
})

function drawNetwork() {
    let v = routing(u, out_capsule_count, out_capsule_dimension);
    addCapsules(u.arraySync(), layer);
    const layer1 = d3.select("#layer1");
    addCapsules(v.arraySync(), layer1);

    //draw links
    d3.select("svg").selectAll("path").remove();
    let in_count = u.shape[0];
    console.log("in count: " + in_count);
    for (let i = 0; i < in_count; i++) {
        for (let j = 0; j < out_capsule_count; j++) {
            drawLink(i, j)
        }
    }
}

function removeCapsule(tensor) {
    let in_capsule_count = tensor.shape[0];
    let in_capsule_dimension = tensor.shape[1];

    const newTensor = tf.slice(tensor, [0, 0], [in_capsule_count - 1, in_capsule_dimension]);

    return newTensor
}

drawNetwork();

// try linking
function drawLink(fromCapsule, toCapsule) {

    const svg = d3.select("svg");

    const width = svg.node().getBoundingClientRect().width;

    const spreadDist = 10
    const fromOffset = d3.scaleLinear([0, u.shape[0] - 1], [-spreadDist/2, spreadDist/2]);
    const toOffset = d3.scaleLinear([0, out_capsule_count - 1], [-spreadDist/2, spreadDist/2]);

    // Define two points
    const source = { x: 0, y: 115 + fromCapsule * 45 + toOffset(toCapsule) };  // Start point
    const target = { x: width, y: 115 + toCapsule * 45 + fromOffset(fromCapsule)}; // End point

    // Create a link generator
    const link = d3.linkHorizontal()
        .x(d => d.x)
        .y(d => d.y);

    // Append the path
    svg.append("path")
        .attr("d", link({ source, target }))
        .attr("fill", "none")
        .attr("stroke", "black")
        .attr("stroke-width", 2);

}