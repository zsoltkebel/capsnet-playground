import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { marginLoss, reconstructionLoss } from "../model/capsnet/trainer";
import { renderImageFromData, updateDigitCaps, updateDynamicRoutingLinks, visualiseBatch, visualisePrediction } from "../model/visualise";
import { CapsuleNetwork, DigitCaps } from "../model/capsnet/capsnet-tensorflow";
import { MNISTDataset } from "../model/capsnet/dataset";
import { QueryableWorker } from "../model/web-worker/queryable-worker";

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
const saveModelCheckbox = d3.select("#save-model");

let currentImageIdx;

let state = {
    coeffs: null,
}


// Handle input image click
d3.select("canvas#input-image")
    .on("click", (event) => {
        // predictRandom();
        modelTrainingTask.sendQuery("predictRandom");
        // webWorker.postMessage({ type: "predict" });
    });

// Handle train button click
trainButton
    .on("click", async (event) => {
        const saveModel = saveModelCheckbox.node().checked;
        console.log("save model?", saveModel);
        // webWorker.postMessage({ type: "start_training" });
        modelTrainingTask.sendQuery("trainModel", { saveModelToBrowserCache: saveModel }); //TODO could send additional params here like epoch, batch size etc

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

const modelTrainingTask = new QueryableWorker(new URL('../model/web-worker/model-tasks.js', import.meta.url));

modelTrainingTask.sendQuery("loadModel", "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/capsnet.json");  //TODO pass url based on config

modelTrainingTask.addListener("modelDidLoad", () => {
    d3.select("canvas#input-image").classed("disabled", false);
    d3.select("#btn-train").node().disabled = false;

    modelTrainingTask.sendQuery("predictRandom");
});

modelTrainingTask.addListener("trainingDidStart", (totalBatches) => {
    trainButton.node().disabled = true;
    saveModelCheckbox.node().disabled = true;

    totalBatchesLabel.textContent = totalBatches;
    progressBar.max = totalBatches;
});

modelTrainingTask.addListener("trainingDidFinish", () => {
    trainButton.node().disabled = false;
    saveModelCheckbox.node().disabled = false;
});

modelTrainingTask.addListener("visualiseSample", (data) => {
    if (document.hidden) {
        console.log("Skipping visualisation because page is hidden");
        return;
    }

    requestAnimationFrame(() => {
        visualiseSample(data);
    });
});


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
