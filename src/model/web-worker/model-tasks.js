import * as tf from "@tensorflow/tfjs";
import { mnistGenerator } from "../capsnet/dataset";
import { CapsuleNetwork } from "../capsnet/capsnet-tensorflow";
import { trainModel, marginLoss, reconstructionLoss } from "../capsnet/trainer";

const TRAIN_TEST_SPLIT_INDEX = 55000;  // 0-55000 train 55000-65000 test

let model;
let testDataset = tf.data.generator(() => mnistGenerator(TRAIN_TEST_SPLIT_INDEX)).batch(1).shuffle(100);
let iterator;
testDataset.iterator().then((result) => {
    iterator = result;
});

/**
* Return a single sample of batch of tensors.

* @param {tf.Tensor} tensor Batch of tensors
* @returns {tf.Tensor} A single sample
*/
function getSampleOfBatch(tensor, idx = 0) {
    const begin = [idx].concat(tensor.shape.slice(1).map(() => 0));
    const newShape = [1].concat(tensor.shape.slice(1));
    return tensor.slice(begin, newShape).squeeze(0);
}

const queryableFunctions = {
    async loadModel(url) {
        try {
            model = await tf.loadLayersModel(url);
            const decoder = model.getLayer("Decoder"); //TODO decoder unused

            model.summary();
            // decoder.summary();

            console.log("Successfully loaded model from GitHub");
        } catch (error) {
            model = new CapsuleNetwork();

            if (url) {
                console.warn(`Could not load model from URL: '${url}', reason:\n`, error);
            }
        }

        await model.save('indexeddb://capsnet');
        console.log("Saved model to browser cache at 'indexeddb://capsnet'");

        const config = {
            conv1: {
                filters: model.getLayer("ReLU_Conv1").getConfig().filters,
                kernelSize: model.getLayer("ReLU_Conv1").getConfig().kernelSize,
                strides: model.getLayer("ReLU_Conv1").getConfig().strides,
            },
            primaryCaps: {
                capsuleDimension: model.getLayer("PrimaryCaps").getConfig().capsuleDimension,
                numChannels: model.getLayer("PrimaryCaps").getConfig().numChannels,
                kernelSize: model.getLayer("PrimaryCaps").getConfig().kernelSize,
                strides: model.getLayer("PrimaryCaps").getConfig().strides,
            },
            digitCaps: {
                capsuleDimension: model.getLayer("DigitCaps").getConfig().capsuleDimension,
                numCapsules: model.getLayer("DigitCaps").getConfig().numCapsules,
                routingIterations: model.getLayer("DigitCaps").getConfig().routingIterations,
            },
        }
        reply("modelDidLoad", config);
    },

    async trainModel({epochs = 1, batchSize = 64, saveModelToBrowserCache = true} = {}) {
        const dataset = tf.data.generator(() => mnistGenerator(0, TRAIN_TEST_SPLIT_INDEX)).batch(batchSize);
        const totalBatches = Math.ceil(TRAIN_TEST_SPLIT_INDEX / batchSize); //TODO somehow should get this from dataset

        reply("trainingDidStart", totalBatches);

        try {
            await trainModel(model, dataset, epochs, {
                reconLossWeight: 1,
                callback: (model, batchIdx, images, labels, capsOutputs, reconstructions, coeffs, marginLoss, reconstructionLoss) => {
                    // Pass first instance of batch to visualise

                    const imageArray = getSampleOfBatch(images).mul(255).arraySync().flat();
                    const labelArray = getSampleOfBatch(labels).arraySync();
                    const reconstructionArray = getSampleOfBatch(reconstructions).mul(255).arraySync().flat();
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
                },
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
