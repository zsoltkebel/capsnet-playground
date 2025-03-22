import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { renderImage } from "./model/visualise";
import { QueryableWorker } from "./model/web-worker/queryable-worker";
import { } from "./model/capsnet/capsnet-tensorflow";  // Empty import needed to register custom classes for loading

d3.select("#btn-train").node().disabled = true;

const currentBatchLabel = d3.select("#current-batch").node();
const totalBatchesLabel = d3.select("#total-batches").node();
const progressBar = d3.select("progress").node();

const canvasImage = d3.select("canvas#input-image").node();
const canvasReconstructedImage = d3.select("canvas#reconstruction-image").node();

const marginLossLabel = d3.select("#margin-loss-label span").node();
const reconstructionLossLabel = d3.select("#reconstruction-loss-label span").node();

const inputLabel = d3.select("#input-label .digits").node();
const reconstructionLabel = d3.select("#reconstruction-label .digits").node();

const trainButton = d3.select("#btn-train");
const downloadModelButton = d3.select("#btn-download-model");
const loadPreTrainedModelButton = d3.select("#btn-load-pretrained-model");
const resetModelButton = d3.select("#btn-reset-model");
const saveModelCheckbox = d3.select("#save-model");

// Dynamic routing control components
const TOTAL_ITERATIONS = 3;
const btnPreviousIteration = d3.select("#btn-routing-previous");
const btnNextIteration = d3.select("#btn-routing-next");
const lblCurrentIteration = d3.select("#routing-visible-iteration");
const lblTotalIterations = d3.select("#routing-total-iterations");

// Loading indicator
const loadingIndicator = d3.select("#loading");

// State object stores everything needed to visualise a sample
let state = {
    visibleRoutingIteration: 2,  // 0-(total routing iterations-1)
    selectedDigit: null,
};

// Web worker that handles model loading, training and prediction
const modelTrainingTask = new QueryableWorker(new URL('./model/web-worker/model-tasks.js', import.meta.url));

let svgWidth = d3.select("svg").node().getBoundingClientRect().width;
window.addEventListener("resize", () => {
    console.log("Window resized!");
    svgWidth = d3.select("svg").node().getBoundingClientRect().width;

    visualiseSample(state);
});

// Handle input image click
d3.select("canvas#input-image")
    .on("click", () => {
        modelTrainingTask.sendQuery("predictRandom");
    });

// Handle train button click
trainButton.on("click", async () => {
    const saveModel = saveModelCheckbox.node().checked;

    modelTrainingTask.sendQuery("trainModel", { saveModelToBrowserCache: saveModel }); //TODO could send additional params here like epoch, batch size etc
});

saveModelCheckbox.on("change", (event) => {
    if (!event.target.checked) {
        if (confirm("Are you sure you want to delete model from browser cache?")) {
            // Delete cached model
            tf.io.removeModel('indexeddb://capsnet')
                .then(() => console.log('Model deleted successfully'))
                .catch(err => console.error('Error deleting model:', err));
        } else {
            // Revert button check
            event.target.checked = true;
        }
    }
});

downloadModelButton.on("click", async () => {
    const model = await tf.loadLayersModel("indexeddb://capsnet");
    model.save("downloads://capsnet")
        .then(() => console.log('Model downloaded successfully'))
        .catch(err => console.error('Error downloading model:', err));
});

loadPreTrainedModelButton.on("click", () => {
    if (confirm("Are you sure you want to load the pre-trained model?")) {
        modelTrainingTask.sendQuery("loadModel", "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/epochs-2/capsnet.json");
        loadingIndicator.style("visibility", "visible");
    }
});

resetModelButton.on("click", () => {
    if (confirm("Are you sure you want to reset the model?")) {
        modelTrainingTask.sendQuery("loadModel", ""); // empty URL will reset the model
        loadingIndicator.style("visibility", "visible");
    }
});

// Dynamic routing controls
lblTotalIterations.text(TOTAL_ITERATIONS);
lblCurrentIteration.text(state.visibleRoutingIteration + 1);

btnPreviousIteration.on("click", () => {
    if (state.visibleRoutingIteration > 0) {
        state.visibleRoutingIteration--;
        lblCurrentIteration.text(state.visibleRoutingIteration + 1);
        visualiseSample(state);
    }
});

btnNextIteration.on("click", () => {
    if (state.visibleRoutingIteration < TOTAL_ITERATIONS - 1) {
        state.visibleRoutingIteration++;
        lblCurrentIteration.text(state.visibleRoutingIteration + 1);
        visualiseSample(state);
    }
});

document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state) {
        // Update visualisation when page becomes visible again
        visualiseSample(state);
    }
});

modelTrainingTask.sendQuery("loadModel");

// Update UI when model loads
modelTrainingTask.addListener("modelDidLoad", (config) => {
    d3.select("canvas#input-image").classed("disabled", false);
    d3.select("#btn-train").node().disabled = false;

    modelTrainingTask.sendQuery("predictRandom");

    updateModelConfig(config);
});

// Update UI when model training begins
modelTrainingTask.addListener("trainingDidStart", (totalBatches) => {
    disableControls(true);

    totalBatchesLabel.textContent = totalBatches;
    progressBar.max = totalBatches;
});

// Update UI when model training finished
modelTrainingTask.addListener("trainingDidFinish", () => {
    disableControls(false);
});

// Update UI when a model reports a run
modelTrainingTask.addListener("visualiseSample", (data) => {
    state = {
        ...state,
        ...data,
    };

    if (document.hidden) {
        // Skipping visualisation because page is hidden
        return;
    }

    requestAnimationFrame(() => {
        visualiseSample(state);
        loadingIndicator.style("visibility", "hidden");
    });
});

function disableControls(disabled) {
    trainButton.node().disabled = disabled;
    saveModelCheckbox.node().disabled = disabled;
    loadPreTrainedModelButton.node().disabled = disabled;
    resetModelButton.node().disabled = disabled;
}

/**
* Visualise a single prediction sample.
*
* @param {{batchIdx: number, image: [number], label: [number], capsuleOutputs: [[number]], reconstruction: [number], coeffs: [[[number]]], marginLoss: number, reconstructionLoss: number}} data 
*/
function visualiseSample(data) {
    renderImage(data.image, canvasImage);
    renderImage(data.reconstruction, canvasReconstructedImage);

    const trueLabel = tf.argMax(data.label, -1).arraySync();
    const predictedLabel = tf.argMax(tf.norm(data.capsuleOutputs, "euclidean", -1), -1).arraySync();
    inputLabel.textContent = trueLabel;
    reconstructionLabel.textContent = predictedLabel;

    updateRoutingLinks(data.coeffs[state.visibleRoutingIteration], state.selectedDigit);
    updateDigitCaps(data, data.capsuleOutputs, { upperLayerX: svgWidth - 30 });

    marginLossLabel.textContent = data.marginLoss.toFixed(8);
    reconstructionLossLabel.textContent = data.reconstructionLoss.toFixed(8);

    if (data.batchIdx) {
        currentBatchLabel.textContent = data.batchIdx + 1;
        progressBar.value = data.batchIdx + 1;
    }
}

/**
* 
* @param {[number]} coeffs array of shape: [numCaps, inputNumCaps] containing the coupling coefficients from the dynamic routing calculation
* @param {number} selectedDigit if not null or undefined, only the links routing to this digit will be shown
*/
async function updateRoutingLinks(coeffs, selectedDigit) {
    const numCaps = coeffs.length;
    const numInputCaps = coeffs[0].length;

    const x = svgWidth - 100;

    const linkGen = d3.linkHorizontal()
        .x(d => d.x)
        .y(d => d.y);

    const upperLayerY = index => (index + 1) * (280 / (numCaps + 1));
    const lowerLayerY = index => (index + 1) * (280 / (numInputCaps + 1));

    const links = coeffs.flatMap((sources, targetIdx) => {
        // filter links if a digit is selected
        if (selectedDigit != null && targetIdx !== selectedDigit) {
            return []
        }
        return sources.map((couplingCoefficient, sourceIdx) => ({
            source: { x: 100, y: lowerLayerY(sourceIdx) },  // Needed for link generator
            target: { x: x, y: upperLayerY(targetIdx) },  // Needed for link generator
            coeff: couplingCoefficient,                          // Coupling coefficient
            sourceIdx: sourceIdx,                                // Capsule index in lower layer
            targetIdx: targetIdx,                                // Capsule index in upper layer
        }))
    });

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
            // Scale coefficients to be at least 0.05 for visibility
            return d3.scaleLinear([0, 1], [0.05, 1.0])(d.coeff);
        });

    // Display weights if there is a selected digit
    if (selectedDigit != null) {
        d3.select("#model svg")
            .selectAll("text")
            .data(links)
            .join("text")
            .attr("x", (d) => d.source.x)
            .attr("y", (d) => d.source.y - 5)
            .text((d) => d.coeff.toFixed(4));
    } else {
        d3.select("#model svg")
            .selectAll("text")
            .remove();
    }
}

/**
* 
* @param {*} digitCapsOutput array with shape: [numCaps, dimCaps]
*/
async function updateDigitCaps(state, digitCapsOutput) {
    const x = svgWidth - 80;

    const arr = digitCapsOutput;//TODO unneccessary
    const highestDigit = tf.argMax(tf.norm(digitCapsOutput, "euclidean", -1), -1).squeeze().arraySync();
    // console.log(highestDigit);
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
        .classed("capsule-predicted", (d, i) => highestDigit === i)
        .classed("capsule-selected", (d, i) => state.selectedDigit === i)
        .style("position", "absolute")
        .style("left", `${x}px`)
        .style("top", (d, i) => `${capsuleY(i)}px`)
        .text((d, i) => i);

    capsulesD3
        .on("mouseenter", (event, d) =>
            updateRoutingLinks(state.coeffs[state.visibleRoutingIteration], d.digit)
        )
        .on("mouseleave", (event, d) =>
            updateRoutingLinks(state.coeffs[state.visibleRoutingIteration], state.selectedDigit)
        )
        .on("click", (event, d) => {
            if (state.selectedDigit === d.digit) {
                state.selectedDigit = null;
            } else {
                state.selectedDigit = d.digit;
            }
            capsulesD3.classed("capsule-selected", (d) => state.selectedDigit === d.digit);
            updateRoutingLinks(state.coeffs[state.visibleRoutingIteration], state.selectedDigit)
        });
}

function updateModelConfig(config) {
    d3.select("#conv1-filters").text(config.conv1.filters);
    d3.select("#conv1-kernel-size").text(`${config.conv1.kernelSize.join("x")}`);
    d3.select("#conv1-strides").text(config.conv1.strides[0]);

    const convOutputHeight = Math.floor((28 - config.conv1.kernelSize[0]) / config.conv1.strides[0]) + 1;
    const primaryCapsOutputHeight = Math.floor((convOutputHeight - config.primaryCaps.kernelSize) / config.primaryCaps.strides) + 1;
    d3.select("#primary-caps-channels").text(config.primaryCaps.numChannels);
    d3.select("#primary-caps-capsule-dimension").text(`${config.primaryCaps.capsuleDimension}D`);
    d3.select("#primary-caps-num-conv-units").text(config.primaryCaps.capsuleDimension);
    d3.select("#primary-caps-kernel-size").text(`${config.primaryCaps.kernelSize}x${config.primaryCaps.kernelSize}`);
    d3.select("#primary-caps-strides").text(config.primaryCaps.strides);
    d3.select("#primary-caps-total-caps").text(`${config.primaryCaps.numChannels}x${primaryCapsOutputHeight}x${primaryCapsOutputHeight}`);  // output height and width are the same in all cases (image is always 28x28)
    d3.select("#primary-caps-output-vector-dimension").text(`${config.primaryCaps.capsuleDimension}D`);

    d3.select("#digit-caps-capsule-dimension").text(`${config.digitCaps.capsuleDimension}D`);
}