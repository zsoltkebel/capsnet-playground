import * as tf from "@tensorflow/tfjs";

const URL = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_grayscale_tensor_uint8_uncompressed";
const MNIST_LABELS_URL = 'https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_labels_uint8';

const BATCH_SIZE = 64;
const BUFFER_SIZE = 64;
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
    return tf.tensor(uint8Array, [this.totalSamples, 10]);

    console.log(`Loaded MNIST dataset labels from ${this.labelsUrl}`);
}

async function fetchBatch(startIndex, batchSize) {
    const response = await fetch(URL, {
        headers: { 'Range': `bytes=${startIndex * IMAGE_SIZE}-${(startIndex + batchSize) * IMAGE_SIZE - 1}` }
    });
    
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

async function* mnistGenerator(startIdx = 0, endIdx = 65000, bufferSize = 50) {
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

class MNISTDataset extends tf.data.Dataset {
    constructor({ url = URL, labelsUrl = MNIST_LABELS_URL, batchSize = BATCH_SIZE, totalSamples = TOTAL_SAMPLES, imageSize = IMAGE_SIZE } = {}) {
        super();
        this.url = url;
        this.labelsUrl = labelsUrl;
        this.batchSize = batchSize;
        this.totalSamples = totalSamples;
        this.imageSize = imageSize;
    }
    
    async fetchLabels() {
        // Load MNIST labels
        // This labels file is relatively small and so keeping it all in memory is not too expensive
        const response = await fetch(this.labelsUrl);
        const buffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        this.labels = tf.tensor(uint8Array, [this.totalSamples, 10]);

        console.log(`Loaded MNIST dataset labels from ${this.labelsUrl}`);
    }

    async fetchBatch(startIndex, batchSize) {
        const response = await fetch(this.url, {
            headers: { 'Range': `bytes=${startIndex * this.imageSize}-${(startIndex + batchSize) * this.imageSize - 1}` }
        });
        
        const buffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        
        const samplesInBatch = Math.min(batchSize, this.totalSamples - startIndex);

        // Convert raw bytes to tensor
        const tensor = tf.tensor(uint8Array, [samplesInBatch, this.imageSize])
        .div(255) // Normalize pixel values to [0,1]
        .reshape([samplesInBatch, 28, 28, 1]); // Reshape to (batch, height, width, channels)
        
        return tensor;
    }
    
    async *iterator() {
        if (!this.labels) {
            await this.fetchLabels();
        }

        for (let i = 0; i < this.totalSamples / this.batchSize; i++) {
            const samplesInBatch = Math.min(this.batchSize, this.totalSamples - i * this.batchSize);

            const xs = await this.fetchBatch(i * this.batchSize, this.batchSize);
            yield {
                xs: xs,
                // TODO fix last batch not being exactly 64
                ys: [this.labels.slice([i * this.batchSize, 0], [samplesInBatch, 10]), xs],
            };
        }
    }
    
    async getRandomSample() {
        if (!this.labels) {
            await this.fetchLabels();
        }

        // Generate a random index between 0 and totalSamples - 1
        const randomIndex = Math.floor(Math.random() * this.totalSamples);

        // Calculate which batch the random index falls into
        const batchIndex = Math.floor(randomIndex / this.batchSize);
        const batchStart = batchIndex * this.batchSize;

        // Fetch the batch containing the random index
        const xs = await this.fetchBatch(batchStart, this.batchSize);
        
        // Get the exact sample from the batch
        const sampleIndex = randomIndex - batchStart;
        const sample = xs.slice([sampleIndex, 0], [1, -1]);  // Select 1 sample from the batch

        // Fetch the label for the selected sample
        const label = this.labels.slice([randomIndex, 0], [1, 10]);

        return { xs: sample, ys: label };
    }

    size() {
        return this.totalSamples;
    }
}

export { MNISTDataset, mnistGenerator, getRandom };