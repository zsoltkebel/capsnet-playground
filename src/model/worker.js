import * as tf from "@tensorflow/tfjs";
import { getRandom, MNISTDataset, mnistGenerator } from "./capsnet/dataset";
import { loadModelFromGitHub } from "./capsnet/model-loader";
import { trainModel } from "./capsnet/trainer";
import { CapsuleNetwork } from "./capsnet/capsnet-tensorflow";
// import { visualiseBatch } from "./visualise";

// Message types received
const LOAD_MODEL = "load_model";
const RESET_MODEL = "reset_model";
const START_TRAINING = "start_training";
const PREDICT = "predict";
// Message types sent to main thread
const MODEL_READY = "model_ready";
const VISUALISE_SAMPLE = "visualise_sample";
const TRAINING_DID_START = "training_did_start";
const TRAINING_DID_FINISH = "training_did_finish";

let model;
// const dataset = new MNISTDataset();
const dataset = tf.data.generator(mnistGenerator);


onmessage = async (event) => {
    const { type, data } = event.data;
    
    console.log(`Message received: ${type}, ${data}`);
    
    let x, y;
    switch (type) {
        case LOAD_MODEL:
            await loadModel();
            await dataset.take(1).toArray();
            ({ x, y } = await getRandom());
            y.print();
            predict(x.expandDims(0), y.expandDims(0));
            postMessage({ type: MODEL_READY });
            break;
        case PREDICT:
            ({ x, y } = await getRandom());
            predict(x.expandDims(0), y.expandDims(0));
            break;
        case RESET_MODEL:
            model = new CapsuleNetwork();
            break;
        case START_TRAINING:
            console.log(dataset.size)
            // postMessage({ type: TRAINING_DID_START, data: { totalBatches: Math.ceil(dataset.totalSamples / dataset.batchSize) } })

            await trainModel(model, dataset.batch(64), 1, { callback: async (model, batchIdx, images, labels, capsOutputs, reconstructions, coeffs, marginLoss, reconstructionLoss) => {
                // Pass first instance of batch to visualise
                
                const imageArray = await getSampleOfBatch(images).mul(255).data();
                const labelArray = getSampleOfBatch(labels).arraySync();
                const reconstructionArray = await getSampleOfBatch(reconstructions).mul(255).data();
                const coeffsArray = getSampleOfBatch(coeffs).arraySync();
                const capsuleOutputsArray = getSampleOfBatch(capsOutputs).arraySync();

                // Tensors can't be passed through messages and for visualisation arrays are needed anyways
                postMessage({
                    type: VISUALISE_SAMPLE,
                    data: {
                        batchIdx: batchIdx,
                        image: imageArray,
                        label: labelArray,
                        capsuleOutputs: capsuleOutputsArray,
                        reconstruction: reconstructionArray,
                        coeffs: coeffsArray,
                    }
                })
            }});

            // Save model
            await model.save('indexeddb://capsnet');
            // await model.save('downloads://capsnet');
            // await decoder.save('indexeddb://decoder')
            console.log("Saved model to browser indexeddb");

            // postMessage({ type: TRAINING_DID_FINISH });
            break;
        default:
            console.warn(`Unknown message type received: ${type}, message: ${event.data}`);
    }
    
};

async function predict(imageTensor, labelTensor) {
    const [capsuleOutputs, reconstructions] = model.predict(imageTensor);
    const coeffs = model.getLayer("DigitCaps").couplingCoefficients;

    // const mLoss = marginLoss(y1, o1);
    // const rLoss = reconstructionLoss(y2, o2);
    // const loss = mLoss.add(tf.mul(rLoss, tf.scalar(reconLossWeight)));
    
    // TODO give better names to variables
    
    const imageArray = await imageTensor.mul(255).data();
    const labelArray = labelTensor.arraySync();
    const reconstructionArray = await getSampleOfBatch(reconstructions).mul(255).data();
    const coeffsArray = getSampleOfBatch(coeffs).arraySync();
    const capsuleOutputsArray = getSampleOfBatch(capsuleOutputs).arraySync();

    // Tensors can't be passed through messages and for visualisation arrays are needed anyways
    postMessage({
        type: VISUALISE_SAMPLE,
        data: {
            image: imageArray,
            label: labelArray,
            capsuleOutputs: capsuleOutputsArray,
            reconstruction: reconstructionArray,
            coeffs: coeffsArray,
        }
    })
}

async function loadModel() {
    const [capsnet, decoder] = await loadModelFromGitHub();
    model = capsnet;

    capsnet.summary();
    decoder.summary();
}

/**
* Return first element of batch of tensors
* @param {*} tensor 
* @returns 
*/
function getSampleOfBatch(tensor, idx = 0) {
    const begin = [idx].concat(tensor.shape.slice(1).map(() => 0));
    const newShape = [1].concat(tensor.shape.slice(1));
    return tensor.slice(begin, newShape).squeeze(0);
}