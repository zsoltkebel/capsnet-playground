import * as tf from "@tensorflow/tfjs";
import * as d3 from "d3";
import { MNISTData } from "./mnist_data";
import { createCapsNet, marginLoss, reconstructionLoss, customAccuracy, reconstructionLoss, DigitCaps } from "./sequential";
import { updateDigitCaps, updateDynamicRoutingLinks } from "./visualise";

d3.select("#btn-train").node().disabled = true;

const currentBatchLabel = d3.select("#current-batch").node();
const totalBatchesLabel = d3.select("#total-batches").node();
const progressBar = d3.select("progress").node();

const canvasInput = d3.select("canvas#input-image").node();
const canvasReconstruction = d3.select("canvas#reconstruction-image").node();

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
loadModelFromGitHub().then((models) => {
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
    
    // console.log(reconstruction.shape);
    // reconstruction.print();
    const capsules = capsout.slice([0, 0, 0], [1, 10 ,16]).squeeze(0);
    const reconstruction = reconstructions.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255));
    
    displayInputAndReconstruction(image, label, reconstruction, labelDigitFromCapsules(capsules));
    
    const mLoss = marginLoss(labelOneHot.expandDims(0), capsout).arraySync().toFixed(8);
    const rLoss = reconstructionLoss(image.expandDims(0), reconstructions).arraySync().toFixed(8);
    
    d3.select("#margin-loss-label span").text(mLoss);
    d3.select("#reconstruction-loss-label span").text(rLoss);
    
    updateDigitCaps(capsnet, capsout);
}

export function labelDigitFromCapsules(capsules) {
    return labelDigitFromOneHot(tf.norm(tf.cast(capsules, 'float32'), "euclidean", -1));
}

export function labelDigitFromOneHot(oneHotVector) {
    return tf.argMax(oneHotVector, 0).squeeze().arraySync();
}

/**
* 
* @param {*} inputImage 
* @param {*} inputLabel 
* @param {*} reconstructedImage 
* @param {*} reconstructedLabel 
*/
async function displayInputAndReconstruction(inputImage, inputDigit, reconstructedImage, reconstructedDigit) {
    renderInput(inputImage, inputDigit);
    renderRecunstruction(reconstructedImage, reconstructedDigit);
}

async function renderInput(image, label) {
    // const canvas = d3.select("canvas#input-image").node();
    await renderImage(image, label, canvasInput);
    d3.select("#input-label .digits").text(label);
}

async function renderRecunstruction(image, label) {
    // const canvas = d3.select("canvas#reconstruction-image").node();
    await renderImage(image, label, canvasReconstruction);
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

async function loadModelFromGitHub() {
    const url = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/capsnet.json";
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
        capsnet.summary();
    } catch (error) {
        console.warn("Failed to load model from browser indexeddb because of the following reason:\n", error);
        // console.error(error);
        
        [capsnet, decoder] = createCapsNet();
    }
    
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
    
    await trainModel(model, trainDataset, 1);
    
    if (d3.select("#save-model").node().checked) {
        // Save model to browser cache
        await model.save('indexeddb://capsnet');
        // await model.save('downloads://capsnet');
        await decoder.save('indexeddb://decoder');
        
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

async function visualizeBatch(batchIdx, xs, y1, y2, o1, o2, marginLoss, reconstructionLoss) {
    updateDigitCaps(capsnet, o1);
    
    const inputImg = xs.slice([0, 0, 0, 0], [1, xs.shape[1], xs.shape[2], xs.shape[3]]).squeeze(0).mul(255);
    const reconImg = o2.slice([0, 0, 0, 0], [1, o2.shape[1], o2.shape[2], o2.shape[3]]).squeeze(0).mul(255);
    
    renderInput(inputImg, labelDigitFromOneHot(y1.slice([0, 0], [1, y1.shape[1]]).squeeze(0)));
    renderRecunstruction(reconImg, labelDigitFromCapsules(o1.slice([0, 0, 0], [1, o1.shape[1], o1.shape[2]]).squeeze(0)));
    
    // Update progress bar
    currentBatchLabel.textContent = batchIdx + 1;
    progressBar.value = batchIdx;
    
    // Update loss labels
    marginLossLabel.textContent = marginLoss.toFixed(8);
    reconstructionLossLabel.textContent = reconstructionLoss.toFixed(8);
}

async function trainModel(model, dataset, epochs, reconLossWeight=1.0) {
    const optimiser = tf.train.adam();
    console.log(model.getLayer("Decoder"));
    
    totalBatchesLabel.textContent = dataset.size;
    progressBar.max = dataset.size;
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        const iterator = await dataset.iterator();
        let result = await iterator.next();
        let batchIndex = 0;  // Initialize batchIndex
        
        while (!result.done) {
            const { xs, ys } = result.value;
            
            // Log batchIndex if needed (or track it for other purposes)
            console.log(`Batch ${batchIndex}`);
            // console.log("xs: ", xs);
            // console.log("ys: ", ys);
            
            await tf.nextFrame();  // Yield to the UI thread
            
            const [y1, y2] = ys;
            
            // Tidy up Tensor operations to avoid memory leaks
            tf.tidy(() => {
                optimiser.minimize(() => {
                    const [o1, o2] = model.apply([xs]);
                    console.log("o1", o1);
                    const mLoss = marginLoss(y1, o1);
                    const rLoss = reconstructionLoss(y2, o2);
                    const loss = mLoss.add(tf.mul(rLoss, tf.scalar(reconLossWeight)));
                    
                    visualizeBatch(batchIndex, xs, y1, y2, o1, o2, mLoss.arraySync(), rLoss.arraySync()); // Pass batchIndex to visualize
                    
                    loss.data().then(l => console.log('Loss', l));
                    return loss;
                });
                console.log(model.getLayer("Decoder").getWeights());
                // Visualize batch after optimization
                // const [o1, o2] = model.predict(xs);
                // visualizeBatch(batchIndex, xs, y1, y2, o1, o2); // Pass batchIndex to visualize
            });
            
            // Move to the next batch and increment batchIndex
            result = await iterator.next();
            batchIndex++;  // Increment the batch index
            
            await tf.nextFrame();  // Yield control back to the UI for responsiveness
        }
    }
}

function predict() {
    const [capsout, reconstruction] = model.predict(data.sample(0).image.expandDims(0));
    console.log(reconstruction.shape);
    reconstruction.print();
    renderRecunstruction(reconstruction.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255)), 8);
}

