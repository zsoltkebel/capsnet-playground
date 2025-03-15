import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { marginLoss, reconstructionLoss } from "../model/capsnet/trainer";
import { renderImageFromData, updateDigitCaps, updateDynamicRoutingLinks, visualiseBatch, visualisePrediction } from "../model/visualise";
import { CapsuleNetwork, DigitCaps } from "../model/capsnet/capsnet-tensorflow";
import { MNISTDataset } from "../model/capsnet/dataset";

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
const deleteModelButton = d3.select("#btn-delete-model");
const downloadModelButton = d3.select("#btn-download-model");

let currentImageIdx;

let state = {
    coeffs: null,
}


// Handle input image click
d3.select("canvas#input-image")
.on("click", (event) => {
    // predictRandom();
    webWorker.postMessage({ type: "predict" });
});

// Handle train button click
trainButton
.on("click", async (event) => {
    trainButton.node().disabled = true;
    webWorker.postMessage({ type: "start_training" });
    // onTrainClicked();
    // data.loadFileInBatches();
    
    // data.saveGrayscaleDatasetToFile();
    // data.fetchBatch(1, 64);
    
    // t = await data.loadCompressedTensorFromUint8GitHub();
    // console.log("shape: ", t.shape);
    // console.log(t);
});

deleteModelButton.on("click", async (event) => {
    if (confirm("Are you sure you want to delete cached model?")) {
        await tf.io.removeModel('indexeddb://capsnet')
        .then(() => console.log('Model deleted successfully'))
        .catch(err => console.error('Error deleting model:', err));
    }
});

downloadModelButton.on("click", async (event) => {
    const model = await tf.loadLayersModel("indexeddb://capsnet");
    model.save("downloads://capsnet")
        .then(() => console.log('Model downloaded successfully'))
        .catch(err => console.error('Error downloading model:', err));
});


const webWorker = new Worker(new URL('../model/worker.js', import.meta.url), {type: 'module'});
webWorker.postMessage({ type: "load_model" });
webWorker.onmessage = (event) => {
    const { type, data } = event.data;
    
    switch (type) {
        case "model_ready":
            visualiseModelParameters(data);

            d3.select("canvas#input-image").classed("disabled", false);
            d3.select("#btn-train").node().disabled = false;
            //TODO model ready flag
            break;
        case "training_did_start":
            const { totalBatches } = data;
            totalBatchesLabel.textContent = totalBatches;
            progressBar.max = totalBatches;
            break;
        case "visualise_sample":
            console.log("visualising sample")
            visualiseSample(data);
            break;
        default:
            console.warn(`Received unknown message type from worker: ${type}`);
    }
};

/**
 * Visualise a single prediction sample
 * @param {*} param0 
 */
function visualiseSample(data) {
    // Save current state
    state = data;

    renderImageFromData(data.image, 0, canvasImage);
    renderImageFromData(data.reconstruction, 0, canvasReconstructedImage);
    
    const trueLabel = tf.argMax(data.label, -1).arraySync();
    const predictedLabel = tf.argMax(tf.norm(data.capsuleOutputs, "euclidean", -1), -1).arraySync();
    inputLabel.textContent = trueLabel;
    reconstructionLabel.textContent = predictedLabel;

    updateDynamicRoutingLinks(data.coeffs);
    updateDigitCaps(data, data.capsuleOutputs);

    if (data.batchIdx) {
        currentBatchLabel.textContent = data.batchIdx + 1;
        progressBar.value = data.batchIdx + 1;
    }
}

function visualiseModelParameters() {
    //TODO to be implemented
}
