import * as tf from "@tensorflow/tfjs";

/**
 * Calculate the margin loss as described in the Dynamic Routing Between Capsules paper.
 * 
 * @param {tf.Tensor} yTrue True labels with shape [batchSize, 10]
 * @param {*} yPred The predicted labels in shape [batchSize, 10, outCapsDimension]
 * @param {*} mPlus Parameter of the margin loss calculation, defaults to 0.9
 * @param {*} mMinus Parameter of the margin loss calculation, defaults to 0.1
 * @param {*} lam Parameter of the margin loss calculation, defaults to 0.5
 * @returns {tf.Tensor} A single value tensor with shape []
 */
function marginLoss(yTrue, yPred, mPlus=0.9, mMinus=0.1, lam=0.5) {
    // Compute the length of the capsule output vectors to [batchSize, numCaps]
    const v_c = tf.sqrt(tf.sum(tf.square(yPred), -1))
    
    // Calculate the margin loss components
    const left = tf.square(tf.maximum(tf.scalar(0), tf.scalar(mPlus).sub(v_c)));
    const right = tf.square(tf.maximum(tf.scalar(0), v_c.sub(tf.scalar(mMinus))));
    
    // Combine the margin loss components using the labels
    let margin_loss = yTrue.mul(left).add(tf.scalar(lam).mul(tf.scalar(1).sub(yTrue)).mul(right));
    
    // Sum over capsules to get shape [batch_size] and average over batches
    margin_loss = tf.sum(margin_loss, 1)
    margin_loss = tf.mean(margin_loss)

    return margin_loss
}

/**
 * Calculate the Mean Squared Error (MSE).
 * 
 * @param {tf.Tensor} yTrue True images
 * @param {tf.Tensor} yPred Reconstructed images
 * @returns {tf.Tensor} A single value tensor with shape []
 */
function reconstructionLoss(yTrue, yPred) {
    return yTrue.sub(yPred).square().mean();
}

async function trainModel(model, dataset, epochs, {
    reconLossWeight=0.0005,
    callback = () => {},
} = {}) {
    const optimiser = tf.train.adam();    
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        const iterator = await dataset.iterator();
        let result = await iterator.next();
        let batchIndex = 0;
        
        while (!result.done) {
            const { x, y } = result.value;
            
            await tf.nextFrame();  // Yield to the UI thread
                        
            // Tidy up Tensor operations to avoid memory leaks
            tf.tidy(() => {
                optimiser.minimize(() => {
                    const [digitCapsOutputs, reconstructions] = model.apply(x);

                    const mLoss = marginLoss(y, digitCapsOutputs);
                    const rLoss = reconstructionLoss(x, reconstructions);
                    const loss = mLoss.add(tf.mul(rLoss, tf.scalar(reconLossWeight)));
                    
                    const coeffs = model.getLayer("DigitCaps").couplingCoefficients;

                    // Use values in this batch to invoke the callback
                    callback(model, batchIndex, tf.keep(x), tf.keep(y), tf.keep(digitCapsOutputs), tf.keep(reconstructions), tf.keep(coeffs), mLoss.arraySync(), rLoss.arraySync());
                    
                    return loss;
                });
            });
            
            // Move to the next batch and increment batchIndex
            result = await iterator.next();
            batchIndex++;
            
            await tf.nextFrame();  // Yield control back to the UI for responsiveness
        }
    }
}

export { trainModel, marginLoss, reconstructionLoss };