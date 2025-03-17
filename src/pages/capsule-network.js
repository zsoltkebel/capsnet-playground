import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { renderImageFromData } from "../model/visualise";
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
const loadPreTrainedModelButton = d3.select("#btn-load-pretrained-model");
const resetModelButton = d3.select("#btn-reset-model");
const saveModelCheckbox = d3.select("#save-model");

// Dynamic routing control components
const TOTAL_ITERATIONS = 3;
const btnPreviousIteration = d3.select("#btn-routing-previous");
const btnNextIteration = d3.select("#btn-routing-next");
const lblCurrentIteration = d3.select("#routing-visible-iteration");
const lblTotalIterations = d3.select("#routing-total-iterations");

let state = {
    visibleRoutingIteration: 2,  // 0-(total routing iterations-1)
};  // Object storing all the data needed for the visualisation

let svgWidth = d3.select("svg").node().getBoundingClientRect().width;
window.addEventListener("resize", () => {
    console.log("Window resized!");
    svgWidth = d3.select("svg").node().getBoundingClientRect().width;

    visualiseSample(state);
});

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

loadPreTrainedModelButton.on("click", () => {
    if (confirm("Are you sure you want to load the pre-trained model?")) {
        modelTrainingTask.sendQuery("loadModel", url = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/epochs-2/capsnet.json");
    }
});

resetModelButton.on("click", () => {
    if (confirm("Are you sure you want to reset the model?")) {
        modelTrainingTask.sendQuery("loadModel", url = ""); // empty URL will reset the model
    }
});

// Dynamic routing controls
lblTotalIterations.text(TOTAL_ITERATIONS);
lblCurrentIteration.text(state.visibleRoutingIteration + 1);

btnPreviousIteration.on("click", (event) => {
    if (state.visibleRoutingIteration > 0) {
        state.visibleRoutingIteration--;
        lblCurrentIteration.text(state.visibleRoutingIteration + 1);
        visualiseSample(state);
    }
});

btnNextIteration.on("click", (event) => {
    if (state.visibleRoutingIteration < TOTAL_ITERATIONS - 1) {
        state.visibleRoutingIteration++;
        lblCurrentIteration.text(state.visibleRoutingIteration + 1);
        visualiseSample(state);
    }
});

document.addEventListener("visibilitychange", () => {
    if (!document.hidden && state) {
        console.log("Visibility changed");
        visualiseSample(state);
    }
});

const modelTrainingTask = new QueryableWorker(new URL('../model/web-worker/model-tasks.js', import.meta.url));

modelTrainingTask.sendQuery("loadModel", "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/epochs-2/capsnet.json");  //TODO pass url based on config

modelTrainingTask.addListener("modelDidLoad", (config) => {
    d3.select("canvas#input-image").classed("disabled", false);
    d3.select("#btn-train").node().disabled = false;

    modelTrainingTask.sendQuery("predictRandom");

    updateModelConfig(config);
});

modelTrainingTask.addListener("trainingDidStart", (totalBatches) => {
    disableControls(true);

    totalBatchesLabel.textContent = totalBatches;
    progressBar.max = totalBatches;
});

modelTrainingTask.addListener("trainingDidFinish", () => {
    disableControls(false);
});

modelTrainingTask.addListener("visualiseSample", (data) => {
    state = {
        ...state,
        ...data,
    };

    if (document.hidden) {
        console.log("Skipping visualisation because page is hidden");
        return;
    }

    requestAnimationFrame(() => {
        visualiseSample(state);
    });
});

function disableControls(disabled) {
    trainButton.node().disabled = disabled;
    saveModelCheckbox.node().disabled = disabled;
    loadPreTrainedModelButton.node().disabled = disabled;
    resetModelButton.node().disabled = disabled;
}

/**
 * Visualise a single prediction sample
 * @param {*} param0 
 */
function visualiseSample(data) {
    renderImageFromData(data.image, 0, canvasImage);
    renderImageFromData(data.reconstruction, 0, canvasReconstructedImage);

    const trueLabel = tf.argMax(data.label, -1).arraySync();
    const predictedLabel = tf.argMax(tf.norm(data.capsuleOutputs, "euclidean", -1), -1).arraySync();
    inputLabel.textContent = trueLabel;
    reconstructionLabel.textContent = predictedLabel;

    updateDynamicRoutingLinks(data.coeffs[state.visibleRoutingIteration], { upperLayerX: svgWidth - 60 });
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
* @param {*} cIJ array of shape: [numCaps, inputNumCaps]
* @param {*} param1 
*/
async function updateDynamicRoutingLinks(cIJ, { selectedTargetIdx } = {}) {
    const numCaps = cIJ.length;
    const numInputCaps = cIJ[0].length;

    const x = svgWidth - 100;
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
            source: { x: 100, y: lowerLayerY(sourceIdx) },  // Needed for link generator
            target: { x: x, y: upperLayerY(targetIdx) },  // Needed for link generator
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
        .classed("capsule-selected", (d, i) => highestDigit === i)
        .style("position", "absolute")
        .style("left", `${x}px`)
        .style("top", (d, i) => `${capsuleY(i)}px`)
        .text((d, i) => i);

    capsulesD3
        .on("mouseenter", (event, d) =>
            updateDynamicRoutingLinks(state.coeffs[state.visibleRoutingIteration], { selectedTargetIdx: d.digit })
        )
        .on("mouseleave", (event, d) =>
            updateDynamicRoutingLinks(state.coeffs[state.visibleRoutingIteration])
        );
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