import * as d3 from "d3";
import * as tf from "@tensorflow/tfjs";

/**
* 
* @param {*} cIJ should have shap: [batch_size, numCaps, inputNumCaps]
* @param {*} param1 
*/
export async function updateDynamicRoutingLinks(cIJ, { lowerLayerX=400, upperLayerX=1000, selectedTargetIdx } = {}) {
    const arr = cIJ.arraySync();
    const numCaps = cIJ.shape[1];
    const numInputCaps = cIJ.shape[2];
    
    // console.log(arr)
    // console.log("here")
    
    const linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
    
    const upperLayerY = index => (index + 1) * (280 / (numCaps + 1));  // X positions based on index
    const lowerLayerY = index => (index + 1) * (280 / (numInputCaps + 1)); // Stagger Y positions
    
    const links = arr[0].flatMap((sources, targetIdx) => 
        sources.map((couplingCoefficient, sourceIdx) => ({
        source: { x: lowerLayerX, y: 45 + lowerLayerY(sourceIdx) },  // Needed for link generator
        target: { x: upperLayerX, y: 45 + upperLayerY(targetIdx) },  // Needed for link generator
        coeff: couplingCoefficient,                          // Coupling coefficient
        sourceIdx: sourceIdx,                                // Capsule index in lower layer
        targetIdx: targetIdx,                                // Capsule index in upper layer
    })));
    
    const linksD3 = d3.select("#model svg")
    .select("#links")
    .selectAll("path")
    .data(links)
    .join("path")
    .attr("d", linkGen)
    .attr("fill", "none")
    .attr("stroke", "black")
    .attr("stroke-width", 2);
    
    linksD3
    .transition()
    .attr("opacity", (d) => {
        if (selectedTargetIdx != null && d.targetIdx !== selectedTargetIdx) {
            return 0.0;
        }
        // Scale coefficients to be at least 0.05 for visibility
        return d3.scaleLinear([0, 1], [0.05, 1.0])(d.coeff);
    });
    
}
// updateCapsules(10);

/**
* 
* @param {*} digitCapsOutput tensor with shape: [batch_size, numCaps, dimCaps]
*/
export async function updateDigitCaps(digitCapsOutput, { lowerLayerX=400, upperLayerX=1020, selectedTargetIdx } = {}) {
    const arr = digitCapsOutput.arraySync();
    const highestDigit = tf.argMax(tf.norm(digitCapsOutput, "euclidean", -1), -1).squeeze().arraySync();
    // console.log(highestDigit);
    // TODO hardcoded first batch (0)
    const capsules = arr[0].flatMap((vector, digit) => {
        const tensor = tf.tensor(vector);
        return {
            digit: digit,
            vector: tensor,
            length: tf.norm(tensor).arraySync(),
        }
    });
    
    const capsuleY = index => (index + 1) * (280 / (capsules.length + 1));  // X positions based on index
    
    const capsulesD3 = d3.select("#model")
    .select("#primary-caps")
    .selectAll("div")
    .data(capsules)
    .join("div")
    .attr("id", (d, i) => `primary-capsule-${i}`)
    .attr("class", "capsule")
    
    capsulesD3
    .transition()
    .style("opacity", (d) => d3.scaleLinear([0, 1], [0.1, 1])(d.length))
    
    
    capsulesD3
    .classed("capsule-selected", (d, i) => highestDigit === i)
    .style("position", "absolute")
    .style("left", `${upperLayerX}px`)
    .style("top", (d, i) => `${45 + capsuleY(i)}px`)
    .text((d, i) => i);
    
}

export function updateDigitCapsules(numCaps, { lowerLayerX=400, upperLayerX=1020, selectedTargetIdx } = {}) {
    //TODO maybe use weight matrix or some other tensor to have more info about capsules?
    
    const arr = Array.from({ length: numCaps }, (_, i) => i);
    
    const capsuleY = index => (index + 1) * (280 / (numCaps + 1));  // X positions based on index
    
    d3.select("#model")
    .select("#primary-caps")
    .selectAll("div")
    .data(arr)
    .join()
    .attr("id", (d, i) => `primary-capsule-${i}`)
    .attr("class", "capsule")
    .style("position", "absolute")
    .style("left", `${upperLayerX}px`)
    .style("top", (d, i) => `${45 + capsuleY(i)}px`)
    .text((d, i) => i);
    // .style("line-height", `${CAPSULE_HEIGHT}px`)
    // .style("height", `${CAPSULE_HEIGHT}px`)
    // .style("width", `${CAPSULE_WIDTH}px`)
}
