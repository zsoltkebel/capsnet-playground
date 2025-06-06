import * as tf from '@tensorflow/tfjs';

class Capsule {
    /**
    * Creates a capsule with id and values
    * @param {string} id 
    * @param {tf.Tensor1D} values 
    */
    constructor(id, values) {
        this.id = id;
        this.values = values;
        this.outputLinks = [];
        this.inputLinks = [];
    }
    
    /**
    * Creates a capsule with random values
    * @param {string} id Unique identifier for the capsule
    * @param {number} dimension Number of values/neurons within capsule
    * @return {Capsule} 
    */
    static random(id, dimension) {
        const values = tf.randomNormal([dimension]);
        return new Capsule(id, values);
    }
    
    get dimension() {
        return this.values.size;
    }

    /**
     * Calculates the layer index of this capsule.
     * Counting starts from layer 0 which are all the capsules capsules that have no input links.
     */
    get layerIndex() {
        return this.inputLinks
            .map(link => link.source.layerIndex + 1)
            .reduce((max, current) => Math.max(max, current), 0);
    }

    get valuesSync() {
        return this.values.arraySync();
    }
}

class Link {
    /**
    * 
    * @param {Capsule} source 
    * @param {Capsule} destination 
    * @param {tf.Tensor2D} weights
    */
    constructor(source, destination, weights) {
        this.source = source;
        this.destination = destination;
        this.weight = weights; //W_ij is a matrix of size (dest dim x source dim)
        this.couplingCoefficient = []; // c_ij matrix value for this link (this is an array to show value at each iteration of the dynamic routing)
    }

    static random(source, destination) {
        const weight = tf.randomNormal([destination.dimension, source.dimension]); //W_ij is a matrix of size (dest dim x source dim)

        let link = new Link(source, destination, weight);
        return link;
    }
    
    /**
    * The "prediction vector" for destination capsule given source capsule
    */
    get u_hat() {
        // u_hat_j|i = W_ij x u_i
        return tf.matMul(this.weight, tf.expandDims(this.source.values, 1)).squeeze();
    }
}

/**
* Squashing function for capsule networks as described in "Dynamic Routing Between Capsules"
* along the last dimension
* @param {tf.Tensor} s Tensor to be squashed
* @returns 
*/
export function squash(s) {
    // Compute Euclidean norm: ||s_j||
    const norm = tf.euclideanNorm(s, -1); // Along the last dimension (-1)
    const normSquared = tf.square(norm);
    
    // Compute the squashing factor: ||s_j||^2 / (1 + ||s_j||^2)
    const squashFactor = tf.div(normSquared, tf.add(normSquared, tf.scalar(1.0)));
    
    // Normalise the vector s_j by dividing by its norm: s_j / ||s_j||
    const normalizedS = tf.div(s, norm.expandDims(-1));
    
    // Final squashed vector: squashFactor * normalizedS
    return tf.mul(squashFactor.expandDims(-1), normalizedS);
}

export class CapsuleNetwork {
    /**
    * Create a new capsule network with randomised parameters
    * @param {tf.Tensor2D[]} capsuleLayers 
    * @param {tf.Tensor4D[]} weights 
    */
    constructor(capsuleLayers, weights, dynamicRoutingIterations = 3) {
        const layers = capsuleLayers.length;

        this.network = buildNetworkFromTensors(capsuleLayers, weights);
        this.dynamicRoutingIterations = dynamicRoutingIterations;
    }

    /**
    * Create a new capsule network with randomised parameters
    * @param {number[]} capsuleCounts Represents the number of capsules at each layer
    * @param {number[]} capsuleDimensions Represents the dimensions of the capsules at each layer
    */
    static random(capsuleCounts, capsuleDimensions) {
        if (capsuleCounts.length != capsuleDimensions.length) {
            throw new Error("Capsule counts and dimensions arrays must have the same number of elements");
        }
        
        let capsnet = new CapsuleNetwork([], []);
        capsnet.network = buildNetwork(capsuleCounts, capsuleDimensions);
        return capsnet;
    }
    
    get layerCount() {
        return this.network.length;
    }

    get allLinks() {
        return this.network.map((layer) => {
            return layer.map((capsule) => capsule.outputLinks).flat();
        }).flat();
    }

    linkLayer(linkIndex) {
        let total = 0;
        for (let i = 0; i < this.network.length; i++) {
            total += this.network[i].map(capsule => capsule.outputLinks).flat().length;
            if (linkIndex < total) {
                return i;
            }
        }
        return this.network.length - 1;
    }

    async forward() {
        for (let layer = 0; layer < this.network.length - 1; layer++) {
            let output = this.dynamicRouting(this.dynamicRoutingIterations, layer);
            
            // feed predictions to next capsules
            for (let j = 0; j < this.network[layer + 1].length; j++) {
                // console.log("feed to next");
                // tf.gather(output, j).print();
                this.network[layer + 1][j].values = tf.gather(output, j);
            }
        }
    }
    
    /**
    * 
    * @param {number} r Number of iterations
    * @param {number} l Layer index of capsules to route to the next layer
    * @returns 
    */
    dynamicRouting(r, l) {
        const i_count = this.network[l].length;
        const j_count = this.network[l + 1].length;
        
        let u_hat_2 = tf.stack(this.network[l].map(capsule => {
            return tf.stack(capsule.outputLinks.map(link => link.u_hat));
        }));
        
        console.log("u_hat_2:");
        u_hat_2.print();
        
        let b_ij = tf.zeros([i_count, j_count]);
        console.log("b_ij:");
        b_ij.print();
        
        let v_j;
        for (let iteration = 0; iteration < r; iteration++) {
            console.log(`Iteration ${iteration + 1}.`);
            
            let c_ij = tf.softmax(b_ij);
            console.log("c_ij:");
            c_ij.print();
            
            // save c_ij values into model for visualisation
            let c_ij_array = c_ij.arraySync();
            for (let ci = 0; ci < c_ij_array.length; ci++) {
                for (let li = 0; li < c_ij_array[ci].length; li++) {
                    this.network[l][ci].outputLinks[li].couplingCoefficient.push(c_ij_array[ci][li]);
                }
            }
            
            let s_j = tf.sum(tf.mul(c_ij.expandDims(2), u_hat_2), 0);
            console.log("s_j:");
            s_j.print();
            
            
            v_j = squash(s_j);
            console.log("v_j:");
            v_j.print();
            
            b_ij = tf.add(b_ij, tf.sum(tf.mul(u_hat_2, v_j), 2));
        }
        
        return v_j;
    }
}

/**
 * Get the 2D tensor representing the weight matrix of capsule i in layer l connecting to capsule j in layer l + 1
 * @param {*} W 
 * @param {*} i 
 * @param {*} j 
 * @param {*} i_dimension 
 * @param {*} j_dimension 
 * @returns 
 */
function getWeightMatrixForCapsule(W, i, j, i_dimension, j_dimension) {
    const weightMatrix = tf.slice(W, [i, j, 0, 0], [1, 1, j_dimension, i_dimension]).squeeze();
    return weightMatrix;
}

function buildNetworkFromTensors(capsuleLayers, weights) {
    // Argument validation
    // if (capsuleLayers.length != weights.length + 1) {
    //     throw new Error("There should be one less number of weight tensors as capsule layers");
    // }
    // for (let i = 0; i < weights.length; i++) {
    //     const [iCount, iDimension] = capsuleLayers[i].shape;
    //     const [jCount, jDimension] = capsuleLayers[i + 1].shape;
    //     const [wICount, wJCount, wJDimension, wIDimension] = weights[i].shape;

    //     if (iCount != wICount || iDimension != wIDimension ||
    //         jCount != wJCount || jDimension != wJDimension) {
    //             throw new Error("Capsule layers and weight do not line up");
    //         }
    // }

    const layers = capsuleLayers.length;

    let network = []

    for (let layerIndex = 0; layerIndex < layers; layerIndex++) {
        let layer = [];
        network.push(layer);
        
        const [capsuleCount, capsuleDimension] = capsuleLayers[layerIndex].shape;

        for (let j = 0; j < capsuleCount; j++) {
            const id = `capsule-${layerIndex}-${j}`;
            const values = tf.gather(capsuleLayers[layerIndex], j);
            let capsule = new Capsule(id, values);
            layer.push(capsule);
            
            if (layerIndex >= 1) {
                // Add links between previous and current layer
                const [capsuleCountPreviousLayer, capsuleDimensionPreviousLayer] = capsuleLayers[layerIndex - 1].shape;
                
                for (let k = 0; k < capsuleCountPreviousLayer; k++) {
                    let previousCapsule = network[layerIndex - 1][k];

                    const capsuleWeight = getWeightMatrixForCapsule(weights[layerIndex - 1], k, j, capsuleDimensionPreviousLayer, capsuleDimension);

                    let link = new Link(previousCapsule, capsule, capsuleWeight);
                    previousCapsule.outputLinks.push(link);
                    capsule.inputLinks.push(link);
                }
            }
        }
    }

    return network;
}

/**
* 
* @param {number[]} capsuleCounts 
* @param {number[]} capsuleDimensions 
* @returns 
*/
function buildNetwork(capsuleCounts, capsuleDimensions) {
    
    let network = []
    
    for (let i = 0; i < capsuleCounts.length; i++) {
        let layer = [];
        network.push(layer);
        
        for (let j = 0; j < capsuleCounts[i]; j++) {
            const id = `capsule-${i}-${j}`;
            let capsule = Capsule.random(id, capsuleDimensions[i]);
            layer.push(capsule);
            
            if (i >= 1) {
                // add links to next layer
                for (let k = 0; k < capsuleCounts[i - 1]; k++) {
                    let previousCapsule = network[i - 1][k];
                    let link = Link.random(previousCapsule, capsule);
                    previousCapsule.outputLinks.push(link);
                    capsule.inputLinks.push(link);
                }
            }
        }
    }
    
    return network
}
