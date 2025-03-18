import * as tf from "@tensorflow/tfjs";

const MNIST_IMAGES_URL = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_grayscale_tensor_uint8_uncompressed";
const MNIST_LABELS_URL = 'https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/mnist-dataset/mnist_labels_uint8';

const NUM_IMAGES_IN_MEMORY = 50;
const IMAGE_SIZE = 28 * 28;  // MNIST images are 28x28 pixels
const NUM_IMAGES = 65000;  // Total number of MNIST images

/**
 * Load the entire mnist image labels in the form of one-hot encodings in a uint8 array,
 * i.e. every 10 values represent a one-hot encoding of the label.
 * 
 * @returns {Promise<Uint8Array>} The array of one-hot vectors.
 */
async function fetchLabels() {
    // Load MNIST labels
    // This labels file is relatively small and so keeping it all in memory is not too expensive
    const response = await fetch(MNIST_LABELS_URL);
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

/**
 * Load a batch of mnist images in the form of their grayscale values in a flat uint8 array.
 * 
 * @param {number} startIndex Index of the first image of the batch to return
 * @param {number} numImages Number of images to fetch
 * @returns {Promise<Uint8Array>} Tha images in the form of a flat array of the pixels grayscale values
 */
async function fetchBatchOfImages(startIndex, numImages) {
    const response = await fetch(MNIST_IMAGES_URL, {
        headers: { 'Range': `bytes=${startIndex * IMAGE_SIZE}-${(startIndex + numImages) * IMAGE_SIZE - 1}` },
    });
    
    const buffer = await response.arrayBuffer();
    return new Uint8Array(buffer);
}

/**
 * Return a generator function that iterates through mnist dataset images starting from image at startIdx, until image at endIdx
 * 
 * @param {number} startIdx Index of the first image
 * @param {number} endIdx Index of the last image (exclusive)
 * @param {number} numImagesInMemory Number of images to load into memory, once these have been iterated over, a new batch of the same number of images is loaded into memory.
 */
async function* mnistGenerator(startIdx = 0, endIdx = NUM_IMAGES, numImagesInMemory = NUM_IMAGES_IN_MEMORY) {
    if (endIdx > NUM_IMAGES) {
        throw Error("End Index cannot be larger than total dataset size");
    }

    let labels = await fetchLabels();

    let bufferIdx = 0;
    let images = await fetchBatchOfImages(startIdx, numImagesInMemory);

    for (let i = startIdx; i < endIdx; i++) {
        if (bufferIdx >= numImagesInMemory) {
            bufferIdx = 0;
            images = await fetchBatchOfImages(i, numImagesInMemory);
        }

        const imageStart = bufferIdx * IMAGE_SIZE;
        const imageEnd = (bufferIdx + 1) * IMAGE_SIZE;

        // Divide pixel values by 255 to get tensor of values between 0 and 1
        const x = tf.tensor(images.slice(imageStart, imageEnd), [28, 28, 1]).div(255);
        const y = tf.tensor(labels.slice(i * 10, (i + 1) * 10), [10]);

        bufferIdx++;

        yield { x, y };
    }
}

export { mnistGenerator };