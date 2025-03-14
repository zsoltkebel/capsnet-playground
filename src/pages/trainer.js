import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { MNISTData } from "../model/capsnet/mnist_data";
import { marginLoss, reconstructionLoss } from "../model/capsnet/trainer";
import { updateDigitCaps, updateDynamicRoutingLinks, visualiseBatch, visualisePrediction } from "../model/visualise";
import { CapsuleNetwork, DigitCaps } from "../model/capsnet/capsnet-tensorflow";
import { trainModel } from "../model/capsnet/trainer";

d3.select("#btn-train").node().disabled = true;

const currentBatchLabel = d3.select("#current-batch").node();
const totalBatchesLabel = d3.select("#total-batches").node();
const progressBar = d3.select("progress").node();



const marginLossLabel = d3.select("#margin-loss-label span").node();
const reconstructionLossLabel = d3.select("#reconstruction-loss-label span").node();

const inputLabel = d3.select("#input-label .digits").node();
const reconstructionLabel = d3.select("#reconstruction-label .digits").node();

const trainButton = d3.select("#btn-train");
const deleteModelButton = d3.select("#btn-delete-model");
const downloadModelButton = d3.select("#btn-download-model");

let currentImageIdx;

const data = new MNISTData();
data.load().then(() => {
    predictRandom();
    
    d3.select("canvas#input-image").classed("disabled", false);
    // d3.select("#btn-train").node().disabled = false;
});

let capsnet, decoder;
loadModelFromGitHub()
// loadModel()
.then((models) => {
    [capsnet, decoder] = models;

    // Attach coupling coefficient visualiser to model
    capsnet.getLayer(DigitCaps.className).routingCallback = (cIJs, vJ) => {
        updateDynamicRoutingLinks(cIJs[2], { 
            // selectedTargetIdx: labelDigitFromCapsules(vJ.slice([0, 0, 0], [1, 10 ,16]).squeeze(0)),
        });
    }

    d3.select("#btn-train").node().disabled = false;
    
    // Display model
    updateDynamicRoutingLinks(
        tf.randomNormal([1, 10, 10], 0, 1),
        {
            // selectedTargetIdx: 2,
        }
    );
    updateDigitCaps(null, tf.randomNormal([1, 10, 16], 0, 1));
});


// Handle input image click
d3.select("canvas#input-image")
.on("click", (event) => {
    predictRandom();
});

// Handle train button click
trainButton
.on("click", (event) => {
    onTrainClicked();
});

deleteModelButton.on("click", async (event) => {
    if (confirm("Are you sure you want to delete cached model?")) {
        await tf.io.removeModel('indexeddb://capsnet')
        .then(() => console.log('Model deleted successfully'))
        .catch(err => console.error('Error deleting model:', err));
    }
});

downloadModelButton.on("click", async (event) => {
    await capsnet.save('downloads://capsnet')
    .then(() => console.log('Model downloaded successfully'))
    .catch(err => console.error('Error downloading model:', err));
});

function predictRandom() {
    currentImageIdx = Math.floor(Math.random() * 65000 + 1);
    predictAndVisualise();
}

export function predictAndVisualise() {
    const {image, labelOneHot, label } = data.sample(currentImageIdx);
    
    // Predict
    const [capsout, reconstructions] = capsnet.predict(image.expandDims(0));    
    
    visualisePrediction(capsnet, image, labelOneHot, capsout.slice([0, 0, 0], [1, capsout.shape[1], capsout.shape[2]]).squeeze(0), reconstructions.slice([0, 0, 0], [1, reconstructions.shape[1], reconstructions.shape[2]]).squeeze(0));

    // // console.log(reconstruction.shape);
    // // reconstruction.print();
    // const capsules = capsout.slice([0, 0, 0], [1, 10 ,16]).squeeze(0);
    // const reconstruction = reconstructions.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255));
    
    // displayInputAndReconstruction(image, label, reconstruction, labelDigitFromCapsules(capsules));
    
    // const mLoss = marginLoss(labelOneHot.expandDims(0), capsout).arraySync().toFixed(8);
    // const rLoss = reconstructionLoss(image.expandDims(0), reconstructions).arraySync().toFixed(8);
    
    // d3.select("#margin-loss-label span").text(mLoss);
    // d3.select("#reconstruction-loss-label span").text(rLoss);
    
    // updateDigitCaps(capsnet, capsout);
}


async function loadModelFromGitHub() {
    const url = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/capsnet.json";
    try {
        const capsnet = await tf.loadLayersModel(url);
        const decoder = capsnet.getLayer("Decoder");
        
        console.log("Successfully loaded model from GitHub");
        return [capsnet, decoder];
    } catch (error) {
        console.warn("Could not load model from GitHub: ", error);
    }
}
async function loadModel() {
    let capsnet, decoder;
    try {
        capsnet = await tf.loadLayersModel('indexeddb://capsnet');
        decoder = capsnet.getLayer("Decoder");
        console.log("Loaded CapsNet from browser cache");
    } catch (error) {
        console.warn("Failed to load model from browser indexeddb because of the following reason:\n", error);
        // console.error(error);
        capsnet = new CapsuleNetwork();
        decoder = capsnet.decoder;
    }
    capsnet.summary();
    decoder.summary();
    
    // Assign callback that displays coupling coefficients whenever the model predicts
    capsnet.getLayer(DigitCaps.className).routingCallback = (cIJs, vJ) => {
        updateDynamicRoutingLinks(cIJs[2], { 
            // selectedTargetIdx: labelDigitFromCapsules(vJ.slice([0, 0, 0], [1, 10 ,16]).squeeze(0)),
        });
    }
    
    return [capsnet, decoder]
}

async function onTrainClicked() {
    // const savedDecoder = await tf.loadLayersModel('indexeddb://decoder');
    
    const model = capsnet;
    // const [model, decoder] = createCapsNet({decoder: savedDecoder});
    // let [model, decoder] = createCapsNet();
    const BATCH_SIZE = 64;
    const BATCHES = 1;
    
    const { trainDataset, testDataset } = data.createDataset(BATCH_SIZE, 65000, 0.8);
    
    trainButton.node().disabled = true;
    
    console.log(trainDataset.size);
    
    console.log("Model output names:", model.outputNames);

    totalBatchesLabel.textContent = trainDataset.size;
    progressBar.max = trainDataset.size;

    await trainModel(model, trainDataset, 1, { callback: visualiseBatch});
    
    if (d3.select("#save-model").node().checked) {
        // Save model to browser cache
        await model.save('indexeddb://capsnet');
        // await model.save('downloads://capsnet');
        // await decoder.save('indexeddb://decoder');
        
        console.log("Saved model to browser indexeddb");
    }
    trainButton.node().disabled = false;
    
    return;
    
    model.compile({
        optimizer: tf.train.adam(),
        loss: {
            'DigitCaps': (yTrue, yPred) => {
                //TODO change so quick
                // console.log(yTrue);
                // console.log(yPred);
                // const input = labelDigitFromOneHot(yTrue.slice([0, 0], [1, yTrue.shape[1]]).squeeze(0));
                // const predicted = labelDigitFromCapsules(yPred.slice([0, 0, 0], [1, yPred.shape[1], yPred.shape[2]]).squeeze(0));
                
                
                // // console.log(input);
                // inputLabel.textContent = input;
                // reconstructionLabel.textContent = predicted;
                
                const loss = marginLoss(yTrue, yPred);
                
                marginLossLabel.textContent = loss.arraySync().toFixed(8);
                
                return loss;
            }, // Custom margin loss
            'Decoder': (yTrue, yPred) => {
                const batch_size = yTrue.shape[0]
                // for (let i = 0; i < batch_size; i++) {
                const img = yTrue.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255));
                renderInput(img, 8);
                
                const rec = yPred.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255));
                renderRecunstruction(rec, 8);
                
                // }
                
                const loss = reconstructionLoss(yTrue, yPred);
                
                reconstructionLossLabel.textContent = loss.arraySync().toFixed(8);
                
                return loss
            },
        },
        lossWeights: {
            'DigitCaps': 1.0, // Give full weight to classification loss
            'Decoder': 0.0005, // Lower weight for reconstruction loss original: 0.0005
        },
        metrics: {
            'DigitCaps': [customAccuracy],   // Accuracy for classification output
            'Decoder': ['mse'],
        },
        // metrics: ['accuracy'],
    });
    
    await model.fitDataset(trainDataset, {
        epochs: 1,
        validationData: testDataset, // Optional validation dataset
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs}`, logs);
                console.log(`  Total Loss = ${logs.loss}`);
                console.log(`  Classification Loss = ${logs.digit_caps_DigitCaps2}`);  // Margin loss
                console.log(`  Reconstruction Loss = ${logs.decoder_Decoder1}`);
            },
            onBatchEnd: async (batch, logs) => {
                
                const progressBar = d3.select("progress").node();
                progressBar.value = batch;
                progressBar.max = trainDataset.size;
                
                console.log(`Batch ${batch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.digit_caps_DigitCaps2_customAccuracy}`);
            },
            onTrainBegin: async (logs) => {
                // Disable train button
                d3.select("#btn-train").node().disabled = true;
            },
            onTrainEnd: async (logs) => {
                // Enable train button
                d3.select("#btn-train").node().disabled = false;
            },
        },
    });
    
    // save decoder
    // await decoder.save('localstorage://decoder');
    if (window.confirm("Do you want to save the model?")) {
        await model.save('indexeddb://capsnet');
        // await model.save('downloads://capsnet');
        await decoder.save('indexeddb://decoder');
    }
    // const saveResult = await decoder.save('downloads://decoder');
    // console.log(saveResult);
    // await decoder.save('file:///Users/zsoltkebel/Developer/university_projects/capsnet_demo/decoder');
    
}





function predict() {
    const [capsout, reconstruction] = model.predict(data.sample(0).image.expandDims(0));
    console.log(reconstruction.shape);
    reconstruction.print();
    renderRecunstruction(reconstruction.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255)), 8);
}

