import * as tf from "@tensorflow/tfjs";

const MNIST_IMAGES_URL = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_grayscale_tensor_uint8_uncompressed";
const MNIST_LABELS_URL = 'https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_labels_uint8';

const BUFFER_SIZE = 50;
const IMAGE_SIZE = 28 * 28;  // MNIST images are 28x28 pixels
const TOTAL_SAMPLES = 65000;

let labels;
let buffer;
let bufferStart = 0;

async function getRandom() {
    const randomIdx = Math.floor(Math.random() * (buffer.length / IMAGE_SIZE));

    const x = tf.tensor(buffer.slice(randomIdx * IMAGE_SIZE, (randomIdx + 1) * IMAGE_SIZE), [28, 28, 1]).div(255);
    const y = tf.tensor(labels.slice((bufferStart + randomIdx) * 10, (bufferStart + randomIdx + 1) * 10), [10]);

    return { x, y };
}

async function fetchLabels() {
    // Load MNIST labels
    // This labels file is relatively small and so keeping it all in memory is not too expensive
    const response = await fetch(MNIST_LABELS_URL);
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

async function fetchBatch(startIndex, batchSize) {
    const response = await fetch(MNIST_IMAGES_URL, {
        headers: { 'Range': `bytes=${startIndex * IMAGE_SIZE}-${(startIndex + batchSize) * IMAGE_SIZE - 1}` }
    });
    
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

async function* mnistGenerator(startIdx = 0, endIdx = TOTAL_SAMPLES, bufferSize = BUFFER_SIZE) {
    if (endIdx > TOTAL_SAMPLES) {
        throw Error("End Index cannot be larger than total dataset size");
    }
    
    labels = await fetchLabels();

    let bufferIdx = 0;
    buffer = await fetchBatch(0, bufferSize);

    for (let i = startIdx; i < endIdx; i++) {
        if (bufferIdx >= bufferSize) {
            bufferStart = i;
            bufferIdx = 0;
            buffer = await fetchBatch(i, bufferSize);
        }

        const x = tf.tensor(buffer.slice(bufferIdx * IMAGE_SIZE, (bufferIdx + 1) * IMAGE_SIZE), [28, 28, 1]).div(255);
        const y = tf.tensor(labels.slice(i * 10, (i + 1) * 10), [10]);

        bufferIdx++;

        yield { x, y };
    }
}

export { mnistGenerator, getRandom, TOTAL_SAMPLES };