import * as tf from '@tensorflow/tfjs';
import {MNISTData} from "./mnist_data.js"
import * as d3 from "d3";
import { updateDigitCaps, updateDynamicRoutingLinks } from './visualise';
import { labelDigitFromCapsules } from './trainPage.js';

const defaultConfig = {
    "ReLU Conv1": {
        kernelSize: 9,
        strides: 1,
        outChannels: 256,
    },
    "PrimaryCaps": {
        capsuleDim: 8,
        nChannels: 32,
        kernelSize: 9,
        strides: 2,
    },
    "DigitCaps": {
        capsuleNum: 10,
        capsuleDim: 16,
    },
};

function squash(vectors, axis = -1) {
    const squaredNorm = vectors.square().sum(axis, true);
    const scale = squaredNorm.div(squaredNorm.add(1));
    return scale.mul(vectors).div(squaredNorm.add(1e-9).sqrt());
}

function softmax(x, axis = 1) {
    const transposedInp = x.transpose([...Array(x.rank - 1).keys()].filter(i => i !== axis).concat([axis]));
    const softmaxed = tf.softmax(transposedInp.reshape([-1, transposedInp.shape[transposedInp.rank - 1]]), axis);
    return softmaxed.reshape(transposedInp.shape).transpose([...Array(x.rank - 1).keys()].concat([axis]));
}

function dynamicRouting(uHat, iterations = 3, { callback }) {
    const batchSize = uHat.shape[0];
    const numCaps = uHat.shape[1];
    const inputNumCaps = uHat.shape[2];
    
    let bIJ = tf.zeros([batchSize, numCaps, inputNumCaps, 1]);
    let cIJs = [];
    let vJ;
    
    for (let iteration = 0; iteration < iterations; iteration++) {
        const transposedLogits = bIJ.transpose([0, 2, 3, 1]);  // Have to transpose tensor so that capsule dimension is last because osftmax only supports that
        const softmaxOutput = tf.softmax(transposedLogits, -1);  // Softmax along the capsule dimension
        const cJI = softmaxOutput.transpose([0, 3, 1, 2]);
        
        cIJs.push(cJI.squeeze(-1));
        
        const sJ = cJI.mul(uHat).sum(2, true);  // Weighted sum
        vJ = squash(sJ);  // vJ.shape: [batchSize, numCaps, 1, dimCaps]
        
        if (iteration < iterations - 1) {
            const agreement = uHat.mul(vJ).sum(-1, true);  // [batchSize, numCaps, inputNumCaps]
            bIJ = bIJ.add(agreement);  // Update coupling coefficients
        }
    }
    
    vJ = vJ.squeeze([2]);
    
    try {
        callback(cIJs, vJ);
    } catch (error) {
        console.warn("Could not call dynamic routing callback: ", error);
    }
    
    return [vJ, ...cIJs];
}

class PrimaryCaps extends tf.layers.Layer {
    static className = "PrimaryCaps";
    
    constructor({
        capsuleDimension = 8,
        numChannels = 1,
        kernelSize = 9,
        strides = 4,
        ...options
    }) {
        super(options);
        
        this.capsuleDimension = capsuleDimension;
        this.numChannels = numChannels;
        this.kernelSize = kernelSize;
        this.strides = strides;        
    }
    
    build(inputShape) {
        this.convLayer = tf.layers.conv2d({
            filters: this.capsuleDimension * this.numChannels,
            kernelSize: this.kernelSize,
            strides: this.strides,
            padding: 'valid',
            kernelInitializer: 'glorotUniform',
        });
        
        this.convLayer.build(inputShape);
        // Register the child layer's weights as part of this layer.
        this.trainableWeights = this.trainableWeights.concat(this.convLayer.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.convLayer.nonTrainableWeights);
        
        // super.build(inputShape);
    }
    
    computeOutputShape(inputShape) {
        // Calculate output shape's height and width
        const height = Math.floor((inputShape[1] - this.kernelSize) / this.strides) + 1;
        const width = Math.floor((inputShape[2] - this.kernelSize) / this.strides) + 1;
        return [null, height * width * this.numChannels, this.capsuleDimension];
    }
    
    call(input) {
        return tf.tidy(() => {
            let capsuleOutputs = this.convLayer.apply(input);
            
            const height = capsuleOutputs.shape[1];
            const width = capsuleOutputs.shape[2];
            
            capsuleOutputs = capsuleOutputs.reshape([-1, height * width * this.numChannels, this.capsuleDimension])
            
            return capsuleOutputs;
        });
    }

    getConfig() {
        return {
            ...super.getConfig(),
            capsuleDimension: this.capsuleDimension,
            numChannels: this.numChannels,
            kernelSize: this.kernelSize,
            strides: this.strides,
        }
    }
}

class DigitCaps extends tf.layers.Layer {
    static className = "DigitCaps";
    
    /**
    * Constructor for the DigiCaps layer
    * 
    * @param {number} numCapsules - Number of capsules, for mnist there are always 10 "classes" (digits 0-9)
    * @param {number} capsuleDimension - Dimensionality of the capsules, defaults to 16
    * @param {number} routingIterations - Number of routing iterations to do when routing primarycaps to digitcaps
    * @param {function([Tensor]): void} routingCallback - Triggered after routing
    * @param {tf.LayerArgs} options - Any additional options to pass onto layers constructor
    */
    constructor({
        numCapsules = 10,
        capsuleDimension = 16,
        routingIterations = 3,
        routingCallback,
        ...options
    }) {
        super(options);
        
        this.numCapsules = numCapsules;
        this.capsuleDimension = capsuleDimension;
        this.routingIterations = routingIterations;
        this.routingCallback = routingCallback;
    }
    
    computeOutputShape() {
        return [null, this.numCapsules, this.capsuleDimension];
    }
    
    build(inputShape) {
        this.inputNumCapsules = inputShape[1];
        this.inputCapsuleDimension = inputShape[2];
        
        // Weight matrix
        this.W = this.addWeight(
            'weights',
            [this.numCapsules, this.inputNumCapsules, this.capsuleDimension, this.inputCapsuleDimension],
            'float32',
            tf.initializers.randomNormal({}),
        );
    }
    
    call(input) {
        return tf.tidy(() => {
            // TensorFlow.js internally passes the inputs as arrays
            if (Array.isArray(input)) {
                input = input[0];
            }
            
            // Make W and u have compatible shapes for matrix multiplication:
            // W [        1, numCapsules, inputNumCapsules,      capsuleDimension, inputCapsuleDimension]
            // u [batchSize,           1, inputNumCapsules, inputCapsuleDimension,                     1]
            let W = this.W.read().expandDims(0);  
            let u = input.expandDims(1).expandDims(-1);  
            
            // Tiling 5-rank tensor doesnt have gradient implemented yet so
            // this is a workaround to broadcasting issue among the first two dimesnions
            W = tf.ones([u.shape[0], 1, 1, 1, 1]).mul(W); // W becomes [64, 10, 1152, 16, 8]
            u = tf.ones([1, W.shape[1], 1, 1, 1]).mul(u); // u becomes [64, 10, 1152, 8, 1]
            
            const uHat = tf.matMul(W, u).squeeze(-1);
            
            // Apply dynamic routing
            return dynamicRouting(uHat, this.routingIterations, { callback: this.routingCallback });
        });
    }
    
    getConfig() {
        return {
            ...super.getConfig(),
            numCapsules: this.numCapsules,
            capsuleDimension: this.capsuleDimension,
            routingIterations: this.routingIterations,
        };
    }
}


function marginLoss(yTrue, yPred, mPlus=0.9, mMinus=0.1, lam=0.5) {
    // yTrue: [batchSize, 10]
    // yPred: [batchSize, 10, 16]
    // console.log("yTrue", yTrue.shape)
    // console.log("yPred", yPred.shape)
    
    // Compute the length of the capsule output vectors to (batch_size, num_capsules)
    const v_c = tf.sqrt(tf.sum(tf.square(yPred), -1))
    // const v_c = yPred;
    
    // Calculate the margin loss components
    const left = tf.square(tf.maximum(tf.scalar(0), tf.scalar(mPlus).sub(v_c)));
    const right = tf.square(tf.maximum(tf.scalar(0), v_c.sub(tf.scalar(mMinus))));
    // console.log("left " + left)
    // console.log("right " + right)
    
    // const oneHot = tf.oneHot(tf.cast(yTrue, 'int32'), 10);
    // Combine the margin loss components using the labels
    let margin_loss = yTrue.mul(left).add(tf.scalar(lam).mul(tf.scalar(1).sub(yTrue)).mul(right));
    
    // Sum over capsules to get shape (batch_size) and average over batches
    margin_loss = tf.sum(margin_loss, 1)
    margin_loss = tf.mean(margin_loss)
    // console.log(margin_loss.arraySync());
    return margin_loss
}

function reconstructionLoss(yTrue, yPred) {
    
    return yTrue.sub(yPred).square().mean();
}

function customAccuracy(yTrue, yPred) {
    // Convert logits or predictions to the predicted class (argMax for multi-class)
    const v_c = tf.sqrt(tf.sum(tf.square(yPred), -1))
    // const v_c = yPred
    // console.log(yTrue)
    
    // console.log(yPred)
    
    const predictedClasses = tf.argMax(v_c, axis=1); // Predicted class from the model output
    const trueClasses = tf.argMax(yTrue, axis=1); // True class labels (one-hot encoded)
    
    // Compare the predicted class with the true class
    const correctPredictions = predictedClasses.equal(trueClasses); // Returns a boolean tensor
    
    const accuracy = correctPredictions.mean(); // Calculate the mean of correct predictions (accuracy)
    return accuracy;
}

class Norm extends tf.layers.Layer {
    static className = "Norm";
    
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1]];
    }
    
    call(inputs) {
        return tf.norm(inputs[0], "euclidean", -1);
    }
}
/**
* Mask a Tensor of shape [None, numCapsule, dimCapsule] by selecting the capsule with the maximum length.
* All other capsules are set to zero except for the selected.
* The masked Tensor is then flattened.
*/
class Mask extends tf.layers.Layer {
    static className = "Mask";
    
    constructor(config = {}) {
        super(config);
    }
    
    computeOutputShape(inputShape) {
        return [inputShape[0], inputShape[1]*inputShape[2]];
    }
    
    call(inputs) {
        return tf.tidy(() => {
            const capsuleOutput = inputs[0]; 
            
            // Compute the vector lengths (capsule activations) to identify the most likely digit for each sample
            const vectorLengths = tf.norm(capsuleOutput, "euclidean", -1); // Shape: [batchSize, 10]
            
            // Create a one-hot mask based on the predicted class (most probable capsule)
            const mask = tf.oneHot(tf.argMax(vectorLengths, 1), vectorLengths.shape[1]).expandDims(2); // Shape: [batchSize, 10, 1]
            
            // Mask the capsule outputs by selecting the most likely capsule's activations and flatten the tensor
            return tf.layers.flatten().apply(capsuleOutput.mul(mask)); // Shape: [batchSize, 10, 16]
        });
    }
}

/**
* Create the Decoder model according to the description in "Dynamic Linking Between Capsules" paper.
* @param {*} param0 
* @returns 
*/
function createDecoderModelSeq({ capsules=10, dimensions=16, imageSize=28, imageChannels=1 } = {}) {
    //TODO perhaps decoder could get an extra tensor or dimension for true labels of the image??
    return tf.sequential({
        layers: [
            new Mask({ inputShape: [capsules, dimensions], name: "MeskedDigitCaps" }),
            tf.layers.dense({ units: 512, activation: 'relu', name: "FC1_ReLU" }),
            tf.layers.dense({ units: 512 * 2, activation: 'relu', name: "FC2_ReLU" }),
            tf.layers.dense({ units: 28 * 28, activation: 'sigmoid', name: "FC3_Sigmoid" }),
            tf.layers.reshape({ targetShape: [imageSize, imageSize, imageChannels]})
        ],
        name: "Decoder",
    });
}


function createCapsNet({
    primaryCapsOptions = {
        dimensions: 8,
        nChannels: 2,
        kernelSize: 9,
        strides: 4,
    },
    digitCapsOptions = {
        
    },
    decoder=createDecoderModelSeq(),
} = {}) {
    const input = tf.input({ shape: [28, 28, 1] });
    
    // ReLU Conv1 Layer
    const conv1 = tf.layers.conv2d({
        filters: 256,
        kernelSize: 9,
        strides: 1,
        padding: 'valid',
        activation: 'relu',
        name: "ReLU_Conv1",
    }).apply(input);
    
    const primaryCapsReshaped = new PrimaryCaps({
        capsuleDimension: primaryCapsOptions.dimensions,
        numChannels: primaryCapsOptions.nChannels,
        kernelSize: primaryCapsOptions.kernelSize,
        strides: primaryCapsOptions.strides,
        name: "PrimaryCaps",
    }).apply(conv1);
    // const primaryCaps = tf.layers.conv2d({
    //     filters: primaryCapsOptions.dimensions * primaryCapsOptions.nChannels,
    //     kernelSize: primaryCapsOptions.kernelSize,
    //     strides: primaryCapsOptions.strides,
    //     name: "PrimaryCaps",
    //     // padding: 'valid'
    // }).apply(conv1);
    
    // const height = primaryCaps.shape[1];
    // const width = primaryCaps.shape[2];
    
    // const primaryCapsReshaped = tf.layers.reshape({
    //     targetShape: [height * width * primaryCapsOptions.nChannels, primaryCapsOptions.dimensions],
    // }).apply(primaryCaps);
    
    // DigitCaps Layer (Assuming you have this custom layer defined)
    const digitCaps = new DigitCaps({
        numCapsules: 10,  // Number of classes
        capsuleDim: 16,    // Dimensionality of each capsule
        iterations: 3,
        name: "DigitCaps",
    }).apply(primaryCapsReshaped);
    
    // const digitCapsNorm = new Norm().apply(digitCaps);
    
    const reconstructions = decoder.apply(digitCaps);
    
    // Create the model
    const capsnet = tf.model({
        inputs: input,
        outputs: [digitCaps, reconstructions],  // Output two things: the capsule output and the reconstructed image
        name: "CapsNet",
    });
    
    console.log(`Model: ${capsnet.name}`);
    capsnet.summary();
    
    console.log(`Model: ${decoder.name}`);
    decoder.summary();
    
    return [capsnet, decoder];
    
}

// Register custom layer classes for saving/loading
tf.serialization.registerClass(PrimaryCaps);
tf.serialization.registerClass(DigitCaps);
tf.serialization.registerClass(Mask);

async function main() {
    let capsnet, decoder;
    try {
        capsnet = await tf.loadLayersModel('indexeddb://capsnet');
        decoder = capsnet.getLayer("Decoder");
        console.log("Loaded CapsNet from browser cache");
    } catch {
        [capsnet, decoder] = createCapsNet();
    }
    // const savedDecoder = await tf.loadLayersModel('indexeddb://decoder');
    
    const model = capsnet;
    // const [model, decoder] = createCapsNet({decoder: savedDecoder});
    // let [model, decoder] = createCapsNet();
    const BATCH_SIZE = 64;
    const BATCHES = 1;
    const data = new MNISTData();
    await data.load();
    const { trainDataset, testDataset } = data.createDataset(BATCH_SIZE, 30000, 0.8);
    
    console.log(trainDataset.size);
    
    console.log("Model output names:", model.outputNames);
    
    model.compile({
        optimizer: tf.train.adam(),
        loss: {
            'DigitCaps': marginLoss, // Custom margin loss
            'Decoder': reconstructionLoss, // Reconstruction loss
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
        epochs: 2,
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
        },
    });
    
    // save decoder
    // await decoder.save('localstorage://decoder');
    if (window.confirm("Do you want to save the model?")) {
        await model.save('indexeddb://capsnet');
        await model.save('downloads://capsnet');
        await decoder.save('indexeddb://decoder');
    }
    // const saveResult = await decoder.save('downloads://decoder');
    // console.log(saveResult);
    // await decoder.save('file:///Users/zsoltkebel/Developer/university_projects/capsnet_demo/decoder');
    
    const [capsout, reconstruction] = model.predict(data.sample(0).image.expandDims(0));
    console.log(reconstruction.shape);
    reconstruction.print();
    renderRecunstruction(reconstruction.slice([0, 0, 0, 0], [1, 28, 28, 1]).squeeze(0).mul(tf.scalar(255)), 8);
}
// main();

async function renderRecunstruction(image, label) {
    const canvas = d3.select("canvas#recunstruction").node();
    await renderImage(image, label, canvas)
}

async function renderInput(image, label) {
    const canvas = d3.select("canvas#input").node();
    await renderImage(image, label, canvas);
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
async function renderRandomImage(mnistData) {
    // Ensure the data is loaded
    await mnistData.load();
    
    // Get a random index from the dataset
    const randomIndex = Math.floor(Math.random() * mnistData.labels.shape[0]);
    
    // Get the corresponding image tensor
    const { image, oneHot, label } = mnistData.sample(randomIndex);
    console.log("label: " + label);
    console.log(image.shape)
    
    // Convert image tensor to a regular JS array for rendering
    const imageArray = await image.array();
    
    // Create a canvas element in the DOM
    const canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
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

// const mnistData = new MNISTData();
// renderRandomImage(mnistData);

export { createDecoderModelSeq, createCapsNet, marginLoss, reconstructionLoss, customAccuracy, DigitCaps };