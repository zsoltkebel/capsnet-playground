import * as d3 from "d3";
import * as tf from "@tensorflow/tfjs";
import { predictAndVisualise } from "../pages/trainer";
import { CapsuleNetwork, DigitCaps } from "./capsnet/capsnet-tensorflow";
import { marginLoss, reconstructionLoss } from "./capsnet/trainer";

const CANVAS_SIZE = 280;

const canvasImage = d3.select("canvas#input-image").node();
const canvasReconstructedImage = d3.select("canvas#reconstruction-image").node();

const marginLossLabel = d3.select("#margin-loss-label span").node();
const reconstructionLossLabel = d3.select("#reconstruction-loss-label span").node();

const currentBatchLabel = d3.select("#current-batch").node();
const totalBatchesLabel = d3.select("#total-batches").node();
const progressBar = d3.select("progress").node();

let hoveredDigit;
let state = {
    coeffs: null,
};

/**
* 
* @param {*} cIJ array of shape: [numCaps, inputNumCaps]
* @param {*} param1 
*/
async function updateDynamicRoutingLinks(cIJ, { lowerLayerX=400, upperLayerX=1000, selectedTargetIdx, } = {}) {
    const numCaps = cIJ.length;
    const numInputCaps = cIJ[0].length;
    
    // console.log(arr)
    // console.log("here")
    
    const linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
    
    const upperLayerY = index => (index + 1) * (280 / (numCaps + 1));  // X positions based on index
    const lowerLayerY = index => (index + 1) * (280 / (numInputCaps + 1)); // Stagger Y positions
    
    // TODO dont always show last iteration
    const links = cIJ.flatMap((sources, targetIdx) => 
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

/**
* 
* @param {*} digitCapsOutput array with shape: [numCaps, dimCaps]
*/
async function updateDigitCaps(state, digitCapsOutput, { lowerLayerX=400, upperLayerX=1020, selectedTargetIdx } = {}) {
    const arr = digitCapsOutput;//TODO unneccessary
    const highestDigit = tf.argMax(tf.norm(digitCapsOutput, "euclidean", -1), -1).squeeze().arraySync();
    // console.log(highestDigit);
    // TODO hardcoded first batch (0)
    const capsules = arr.flatMap((vector, digit) => {
        const tensor = tf.tensor(vector);
        return {
            digit: digit,
            vector: tensor,
            length: tf.norm(tensor).arraySync(),
        }
    });
    
    const capsuleY = index => (index + 1) * (280 / (capsules.length + 1));  // X positions based on index
    
    const capsulesD3 = d3.select("#model")
    .select("#digit-caps")
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
    
    capsulesD3
    .on("mouseenter", (event, d) => 
        updateDynamicRoutingLinks(state.coeffs[state.visibleRoutingIteration], { selectedTargetIdx: d.digit })
    )
    .on("mouseleave", (event, d) => 
        updateDynamicRoutingLinks(state.coeffs[state.visibleRoutingIteration])
    );
    
    // Add title
    d3.select("#model")
    .select("p")
    .style("position", "absolute")
    .style("left", `${upperLayerX}px`)
    .style("top", "40px")
    .style("transform", "translate(-50%, -50%)")
    .text("DigitCaps");
}

function updatePrimaryCaps(numCaps, { lowerLayerX=400, upperLayerX=1020, selectedTargetIdx } = {}) {
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

async function visualiseBatch(model, batchIdx, xs, y1, y2, o1, o2, marginLoss, reconstructionLoss) {
    updateDigitCaps(model, o1);
    
    const label = y1.slice([0, 0], [1, y1.shape[1]]).squeeze(0);
    const capsuleOut = o1.slice([0, 0, 0], [1, o1.shape[1], o1.shape[2]]).squeeze(0);
    const inputImg = xs.slice([0, 0, 0, 0], [1, xs.shape[1], xs.shape[2], xs.shape[3]]).squeeze(0);
    const reconImg = o2.slice([0, 0, 0, 0], [1, o2.shape[1], o2.shape[2], o2.shape[3]]).squeeze(0);
    
    visualisePrediction(model, inputImg, label, capsuleOut, reconImg);
    
    // Update progress bar
    currentBatchLabel.textContent = batchIdx + 1;
    progressBar.value = batchIdx;
    
    // Update loss labels
    marginLossLabel.textContent = marginLoss.toFixed(8);
    reconstructionLossLabel.textContent = reconstructionLoss.toFixed(8);
}

/**
 * 
 * @param {CapsuleNetwork} model 
 * @param {tf.Tensor} image - 3D tensor with 1 channel
 * @param {tf.Tensor} label - 1D tensor, one-hot encoding of label digit.
 * @param {tf.Tensor} capsuleOutput - 3D tensor with 1 channel
 * @param {tf.Tensor} reconstructedImage - 3D tensor with 1 channel
 */
async function visualisePrediction(model, image, label, capsuleOutput, reconstructedImage) {
    const labelDigit = tf.argMax(label).arraySync();
    const capsuleDigit = tf.argMax(tf.norm(capsuleOutput, 'euclidean', -1)).arraySync();

    updateDigitCaps(model, capsuleOutput);
    
    renderInput(image.mul(255), labelDigit);
    renderRecunstruction(reconstructedImage.mul(255), capsuleDigit);
    
    // Update loss labels
    // marginLossLabel.textContent = marginLoss.toFixed(8);
    // reconstructionLossLabel.textContent = reconstructionLoss.toFixed(8);
}

async function renderInput(image, label) {
    // const canvas = d3.select("canvas#input-image").node();
    await renderImage(image, label, canvasImage);
    d3.select("#input-label .digits").text(label);
}

async function renderRecunstruction(image, label) {
    // const canvas = d3.select("canvas#reconstruction-image").node();
    await renderImage(image, label, canvasReconstructedImage);
    d3.select("#reconstruction-label .digits").text(label);
}

async function renderImage(image, label, canvas) {
    // console.log("label: " + label);
    
    // Convert image tensor to a regular JS array for rendering
    const imageArray = await image.array();
    
    // Create a canvas element in the DOM
    // const canvas = d3.select("canvas").node();
    // const canvas = document.createElement('canvas');
    // document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    ctx.imageSmoothingEnabled = false;
    
    // Set the canvas size to match the image (28x28 pixels)
    canvas.width = 28;
    canvas.height = 28;
    
    // Convert the image array to a format suitable for the canvas (RGBA format)
    const imageData = ctx.createImageData(28, 28);
    const pixels = imageArray.flat();  // Flatten the 2D array to 1D
    
    for (let i = 0; i < pixels.length; i++) {
        imageData.data[i * 4] = pixels[i];     // Red channel (greyscale image)
        imageData.data[i * 4 + 1] = pixels[i]; // Green channel (greyscale image)
        imageData.data[i * 4 + 2] = pixels[i]; // Blue channel (greyscale image)
        imageData.data[i * 4 + 3] = 255;       // Alpha channel (fully opaque)
    }
    
    // Put the image data on the canvas
    ctx.putImageData(imageData, 0, 0);
}

async function renderImageFromData(pixels, label, canvas) {
    // console.log("label: " + label);
    // console.log("image data ", image);
    
    // Create a canvas element in the DOM
    // const canvas = d3.select("canvas").node();
    // const canvas = document.createElement('canvas');
    // document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    ctx.imageSmoothingEnabled = false;
    
    // Set the canvas size to match the image (28x28 pixels)
    canvas.width = 28;
    canvas.height = 28;
    
    // Convert the image array to a format suitable for the canvas (RGBA format)
    const imageData = ctx.createImageData(28, 28);
    // const pixels = image.flat();  // Flatten the 2D array to 1D
    
    for (let i = 0; i < pixels.length; i++) {
        imageData.data[i * 4] = pixels[i];     // Red channel (greyscale image)
        imageData.data[i * 4 + 1] = pixels[i]; // Green channel (greyscale image)
        imageData.data[i * 4 + 2] = pixels[i]; // Blue channel (greyscale image)
        imageData.data[i * 4 + 3] = 255;       // Alpha channel (fully opaque)
    }
    
    // Put the image data on the canvas
    ctx.putImageData(imageData, 0, 0);
}

export { updateDynamicRoutingLinks, updateDigitCaps, updatePrimaryCaps, visualiseBatch, visualisePrediction, renderImageFromData };