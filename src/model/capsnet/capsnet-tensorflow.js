import * as tf from '@tensorflow/tfjs';

class PrimaryCaps extends tf.layers.Layer {
    static className = "PrimaryCaps";
    
    constructor({
        capsuleDimension = 8,
        numChannels = 1,
        kernelSize = 9,
        strides = 4,
        ...options
    } = {}) {
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
        
        // Register the conv layer's weights
        this.trainableWeights = this.trainableWeights.concat(this.convLayer.trainableWeights);
        this.nonTrainableWeights = this.nonTrainableWeights.concat(this.convLayer.nonTrainableWeights);
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

/**
 * Compute dynamic routing.
 * 
 * @param {tf.Tensor} uHat Tensor of shape [batchSize, numOutCaps, numInCaps, outCapsDimension], these are the rediction vectors
 * @param {number} iterations Number of routing iterations to compute
 * @returns {[tf.Tensor, tf.Tensor]} returns capsule outputs in shape [batchSize, numOutCaps, outCapsDimension] and a tensor of coupling coefficients in the shape [batchSize, iterations, numOutCaps, numInCaps]
 */
function dynamicRouting(uHat, iterations = 3) {
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
    
    vJ = vJ.squeeze(2);  // Remove singleton dimension that was necessary for operations to line up
    
    return [vJ, tf.stack(cIJs, 1)];
}

/**
 * Apply the quashing function on a tensor along a given axis, keeping the original tensor's dimensions.
 * 
 * @param {tf.Tensor} vectors The tensor to operate on
 * @param {number} axis The axis along which to compute squash (defaults to -1 which corresponds to the last dimension)
 * @returns {tf.Tensor} The tensor with the squashed dimension
 */
function squash(vectors, axis = -1) {
    const squaredNorm = vectors.square().sum(axis, true);
    const scale = squaredNorm.div(squaredNorm.add(1));
    return scale.mul(vectors).div(squaredNorm.add(1e-9).sqrt());
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
        ...options
    } = {}) {
        super(options);
        
        this.numCapsules = numCapsules;
        this.capsuleDimension = capsuleDimension;
        this.routingIterations = routingIterations;
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
            const [vJ, couplingCoefficients] = dynamicRouting(uHat, this.routingIterations);
            
            // Save coupling coefficient for visualisation
            if (this.couplingCoefficients instanceof tf.Tensor) {
                this.couplingCoefficients.dispose();
            }
            this.couplingCoefficients = tf.keep(couplingCoefficients);
            return vJ
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

const smallConfig = {
    convOptions: { filters: 256, kernelSize: 9, strides: 1 },
    primaryCapsOptions: { capsuleDimension: 8, numChannels: 1, kernelSize: 9, strides: 4},
    digitCapsOptions: { capsuleDimension: 16, numCapsules: 10, routingIterations: 3 },
};

class CapsuleNetwork extends tf.LayersModel {
    constructor({
        convOptions = { filters: 256, kernelSize: 9, strides: 1 },
        primaryCapsOptions = { capsuleDimension: 8, numChannels: 32, kernelSize: 9, strides: 2},
        digitCapsOptions = { capsuleDimension: 16, numCapsules: 10, routingIterations: 3 },
        decoderOptions = { capsuleDimension: 16, numCapsules: 10 },
        ...options
    } = smallConfig) {
        const input = tf.input({ shape: [28, 28, 1] });
        
        // ReLU Conv1 Layer
        const conv1 = tf.layers.conv2d({
            ...convOptions,
            padding: 'valid',
            activation: 'relu',
            name: "ReLU_Conv1",
        }).apply(input);
        
        // PrimaryCaps
        const primaryCapsReshaped = new PrimaryCaps({
            ...primaryCapsOptions,
            name: "PrimaryCaps",
        }).apply(conv1);
        
        // DigitCaps
        const digitCaps = new DigitCaps({
            ...digitCapsOptions,
            name: "DigitCaps",
        }).apply(primaryCapsReshaped);
        
        // const digitCapsNorm = new Norm().apply(digitCaps);
        
        const decoder = createDecoder(decoderOptions);
        const reconstructions = decoder.apply(digitCaps);
        
        // Create the model
        super({
            inputs: input,
            outputs: [digitCaps, reconstructions],  // Output two things: the capsule output and the reconstructed image
            name: "CapsNet",
            ...options,
        });
        
        this.convOptions = convOptions;
        this.primaryCapsOptions = primaryCapsOptions;
        this.digitCapsOptions = digitCapsOptions;
        this.decoderOptions = decoderOptions;

        this.decoder = decoder;
    }
    
    getConfig() {
        return {
            ...super.getConfig(),
            convOptions: this.convOptions,
            primaryCapsOptions: this.primaryCapsOptions,
            digitCapsOptions: this.digitCapsOptions,
            decoderOptions: this.decoderOptions,
        }
    }
}

/**
* Create the Decoder model according to the description in "Dynamic Linking Between Capsules" paper.

* @param {number} numCapsules Number of incoming capsules
* @param {number} capsuleDimension Dimension of incoming capsules
* @param {number} imageSize The number of pixels of image in each direction width and height
* @param {number} imageChannels Number of channels for resulting image
* @returns {tf.Sequential} The decoder model
*/
function createDecoder({ numCapsules = 10, capsuleDimension = 16, imageSize = 28, imageChannels = 1 } = {}) {
    //TODO perhaps decoder could get an extra tensor or dimension for true labels of the image??
    return tf.sequential({
        layers: [
            new Mask({ inputShape: [numCapsules, capsuleDimension], name: "MeskedDigitCaps" }),
            tf.layers.dense({ units: 512, activation: 'relu', name: "FC1_ReLU" }),
            tf.layers.dense({ units: 512 * 2, activation: 'relu', name: "FC2_ReLU" }),
            tf.layers.dense({ units: 28 * 28, activation: 'sigmoid', name: "FC3_Sigmoid" }),
            tf.layers.reshape({ targetShape: [imageSize, imageSize, imageChannels], name: "Reshape" })
        ],
        name: "Decoder",
    });
}

// Register custom layer classes for saving/loading
tf.serialization.registerClass(PrimaryCaps);
tf.serialization.registerClass(DigitCaps);
tf.serialization.registerClass(Mask);
tf.serialization.registerClass(CapsuleNetwork);

export { CapsuleNetwork, squash, dynamicRouting };