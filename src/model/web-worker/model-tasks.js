import * as tf from "@tensorflow/tfjs";
import { mnistGenerator } from "../capsnet/dataset";
import { CapsuleNetwork } from "../capsnet/capsnet-tensorflow";
import { trainModel, marginLoss, reconstructionLoss } from "../capsnet/trainer";

let model;
let testDataset = tf.data.generator(mnistGenerator).batch(1).shuffle(100);
let iterator;
testDataset.iterator().then((result) => {
    iterator = result;
});

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

const queryableFunctions = {
    async loadModel(url = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/capsnet.json") {
        try {
            model = await tf.loadLayersModel(url);
            const decoder = model.getLayer("Decoder"); //TODO decoder unused

            model.summary();
            // decoder.summary();

            console.log("Successfully loaded model from GitHub");
        } catch (error) {
            model = new CapsuleNetwork();

            console.warn(`Could not load model from URL: '${url}', reason:\n`, error);
        }

        reply("modelDidLoad",); //TODO additional info?
    },

    async trainModel({epochs = 1, batchSize = 64, saveModelToBrowserCache = true} = {}) {
        const dataset = tf.data.generator(mnistGenerator).batch(batchSize);
        const totalBatches = Math.ceil(65000 / 64); //TODO somehow should get this from dataset

        reply("trainingDidStart", totalBatches);

        try {
            await trainModel(model, dataset, epochs, {
                callback: async (model, batchIdx, images, labels, capsOutputs, reconstructions, coeffs, marginLoss, reconstructionLoss) => {
                    // Pass first instance of batch to visualise

                    const imageArray = await getSampleOfBatch(images).mul(255).data();
                    const labelArray = getSampleOfBatch(labels).arraySync();
                    const reconstructionArray = await getSampleOfBatch(reconstructions).mul(255).data();
                    const coeffsArray = getSampleOfBatch(coeffs).arraySync();
                    const capsuleOutputsArray = getSampleOfBatch(capsOutputs).arraySync();

                    // Tensors can't be passed through messages and for visualisation arrays are needed anyways
                    reply("visualiseSample", {
                        batchIdx: batchIdx,
                        image: imageArray,
                        label: labelArray,
                        capsuleOutputs: capsuleOutputsArray,
                        reconstruction: reconstructionArray,
                        coeffs: coeffsArray,
                        marginLoss: marginLoss,
                        reconstructionLoss: reconstructionLoss,
                    });
                }
            });
        } finally {
            if (saveModelToBrowserCache) {
                await model.save('indexeddb://capsnet');
                console.log("Saved model to browser cache at 'indexeddb://capsnet'");
            }

            reply("trainingDidFinish",); //TODO additional info?
        }
    },

    async predictRandom() {
        const sample = (await iterator.next()).value;
        const { x, y } = sample;

        const [capsuleOutputs, reconstructions] = model.predict(x);
        const coeffs = model.getLayer("DigitCaps").couplingCoefficients;

        const mLoss = marginLoss(y, capsuleOutputs).arraySync();
        const rLoss = reconstructionLoss(x, reconstructions).arraySync();

        const imageArray = await x.mul(255).data();
        const labelArray = y.arraySync();
        const reconstructionArray = await getSampleOfBatch(reconstructions).mul(255).data();
        const coeffsArray = getSampleOfBatch(coeffs).arraySync();
        const capsuleOutputsArray = getSampleOfBatch(capsuleOutputs).arraySync();

        // Tensors can't be passed through messages and for visualisation arrays are needed anyways
        reply("visualiseSample", {
            image: imageArray,
            label: labelArray,
            capsuleOutputs: capsuleOutputsArray,
            reconstruction: reconstructionArray,
            coeffs: coeffsArray,
            marginLoss: mLoss,
            reconstructionLoss: rLoss,
        });
    },
};

// system functions

function defaultReply(message) {
    console.warn("Do not send messages to this worker using the 'postMessage()' method. Make all calls via QueryableWorker's 'sendQuery()' method");
}

function reply(queryMethodListener, ...queryMethodArguments) {
    if (!queryMethodListener) {
        throw new TypeError("reply - not enough arguments");
    }
    postMessage({
        queryMethodListener,
        queryMethodArguments,
    });
}

onmessage = (event) => {
    if (
        event.data instanceof Object &&
        Object.hasOwn(event.data, "queryMethod") &&
        Object.hasOwn(event.data, "queryMethodArguments")
    ) {
        console.log(`Executing function '${event.data.queryMethod}'`);
        queryableFunctions[event.data.queryMethod].apply(
            self,
            event.data.queryMethodArguments,
        );
    } else {
        defaultReply(event.data);
    }
};
