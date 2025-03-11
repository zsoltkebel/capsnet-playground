import * as tf from "@tensorflow/tfjs"


const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';


export class MNISTData {
    constructor(imagesFilepath=MNIST_IMAGES_SPRITE_PATH, labelsFilepath=MNIST_LABELS_PATH) {
        this.imagesFilepath = imagesFilepath;
        this.labelsFilepath = labelsFilepath;
        this.didFinishLoading = false;
    }
    
    async load() {
        const canvas = document.createElement('canvas');
        this.ctx = canvas.getContext('2d', { willReadFrequently: true });
        
        const img = new Image();
        img.src = this.imagesFilepath;
        img.crossOrigin = '';
        
        // Load MNIST labels
        const response = await fetch(this.labelsFilepath);
        const buffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(buffer);
        this.labels = tf.tensor(uint8Array, [65000, 10]);
        
        // Load MNIST images sprite
        await new Promise((resolve, reject) => {
            img.onload = () => {
                // Set canvas size to the size of the image
                canvas.width = img.width;
                canvas.height = img.height;
                // Draw the image onto the canvas
                this.ctx.drawImage(img, 0, 0);
                
                resolve();
            };
            
            img.onerror = (err) => {
                console.error("Image loading failed: ");
                // Reject the promise if there's an error loading the image
                reject(`Error loading image: ${err}`);
            };
        });
        
        this.didFinishLoading = true;
    }
    
    checkIfLoaded() {
        if (!this.didFinishLoading) {
            throw new Error(".load() needs to be called before being able to create dataset.");
        }
    }
    
    sample(index) {
        this.checkIfLoaded();
        
        const labelOneHot = this.labels.slice([index, 0], [1, this.labels.shape[1]]).squeeze(0);
        
        return {
            image: this.getImageTensor(index),
            labelOneHot: labelOneHot,
            label: tf.argMax(labelOneHot).arraySync(),
        };
    }
    
    getImageTensor(index) {
        const pixels = this.getImage(index);
        const pixelsTensor = tf.tensor(pixels, [28, 28, 4]);
        // Since the images are greyscale just take the red value (first value in the tuple of 4)
        return tf.slice(pixelsTensor, [0, 0, 0], [28, 28, 1]);
    }
    
    getImage(index) {
        return this.getPixels(0, index, 28*28, 1);
    }
    
    getPixels(x, y, width, height) {
        this.checkIfLoaded();
        
        // Get the pixel data for the region you want (e.g., 10x10 pixels starting from (x, y))
        const imageData = this.ctx.getImageData(x, y, width, height);
        return imageData.data;
    }
    
    createDataset(batchSize, total, trainTestSplit) {
        this.checkIfLoaded();

        const trainSize = Math.floor(trainTestSplit * total);
        
        const trainDataset = this._createDataset(batchSize, 0, trainSize); // Create training dataset
        const testDataset = this._createDataset(batchSize, trainSize, total); // Create testing dataset
        
        return { trainDataset, testDataset };
    }

    _createDataset(batchSize, startIdx, endIdx) {
        this.checkIfLoaded();
        
        // The generator function yields data samples one by one in batches
        const generator = async function* (batchSize) {
            let index = startIdx;
            while (index < endIdx) {
                const batchImages = [];
                const batchLabels = [];
                
                for (let i = 0; i < batchSize; i++) {
                    if (index >= endIdx) break; // Ensure we don't exceed the total dataset size
                    
                    const { image, labelOneHot, label } = this.sample(index);
                    const normalizedImage = image.div(tf.scalar(255.0));

                    batchImages.push(normalizedImage);
                    batchLabels.push(labelOneHot); //TODO might need the integer label instead of onehot encoding
                    index++;
                }
                
                // Stack the batch of images and labels
                const imageTensor = tf.stack(batchImages);
                const labelTensor = tf.stack(batchLabels);

                yield { xs: imageTensor, ys: [labelTensor, imageTensor] };
            }
        };
        
        // Create a dataset from the generator
        const dataset = tf.data.generator(() => generator.call(this, batchSize));
        
        return dataset;
    }
}


async function main() {
    const d = new MNISTData()
    await d.load();
    const dataset = await d.createDataset(64);
    
    dataset.take(3).forEachAsync(({ xs, ys }) => {
        console.log("xs: " + xs.shape);
        console.log("ys: " + ys.shape);
        // xs.print();
        // ys.print();
    });
    
    // const dataset = await createDataset(64);
    
    // dataset.take(1).forEachAsync(batch => {
        //     console.log('Batch images:', batch.xs.shape);
    //     console.log('Batch labels:', batch.ys.shape);
    // });
}
// main()

