import * as tf from "@tensorflow/tfjs";

function marginLoss(yTrue, yPred, mPlus=0.9, mMinus=0.1, lam=0.5) {
    // yTrue: [batchSize, 10]
    // yPred: [batchSize, 10, 16]
    // console.log("yTrue", yTrue.shape)
    // console.log("yPred", yPred.shape)
    
    // Compute the length of the capsule output vectors to (batch_size, num_capsules)
    const v_c = tf.sqrt(tf.sum(tf.square(yPred), -1))
    // const v_c = yPred;
    
    // Calculate the margin loss components
    const left = tf.square(tf.maximum(tf.scalar(0), tf.scalar(mPlus).sub(v_c)));
    const right = tf.square(tf.maximum(tf.scalar(0), v_c.sub(tf.scalar(mMinus))));
    // console.log("left " + left)
    // console.log("right " + right)
    
    // const oneHot = tf.oneHot(tf.cast(yTrue, 'int32'), 10);
    // Combine the margin loss components using the labels
    let margin_loss = yTrue.mul(left).add(tf.scalar(lam).mul(tf.scalar(1).sub(yTrue)).mul(right));
    
    // Sum over capsules to get shape (batch_size) and average over batches
    margin_loss = tf.sum(margin_loss, 1)
    margin_loss = tf.mean(margin_loss)
    // console.log(margin_loss.arraySync());
    return margin_loss
}

/**
 * Mean Squared Error (MSE)
 * @param {*} yTrue 
 * @param {*} yPred 
 * @returns 
 */
function reconstructionLoss(yTrue, yPred) {
    return yTrue.sub(yPred).square().mean();
}

async function trainModel(model, dataset, epochs, { reconLossWeight=1.0, callback } = {}) {
    const optimiser = tf.train.adam();    
    
    for (let epoch = 0; epoch < epochs; epoch++) {
        const iterator = await dataset.iterator();
        let result = await iterator.next();
        let batchIndex = 0;  // Initialize batchIndex
        
        while (!result.done) {
            const { x, y } = result.value;
            
            // console.log(`Batch ${batchIndex}`);
            // console.log("xs: ", xs);
            // console.log("ys: ", ys);
            
            await tf.nextFrame();  // Yield to the UI thread
            
            // const [y1, y2] = ys;
            
            // Tidy up Tensor operations to avoid memory leaks
            tf.tidy(() => {
                optimiser.minimize(() => {
                    const [o1, o2] = model.apply(x);

                    const mLoss = marginLoss(y, o1);
                    const rLoss = reconstructionLoss(x, o2);
                    const loss = mLoss.add(tf.mul(rLoss, tf.scalar(reconLossWeight)));
                    
                    // TODO give better names to variables
                    const coeffs = model.getLayer("DigitCaps").couplingCoefficients;

                    callback(model, batchIndex, tf.keep(x), tf.keep(y), tf.keep(o1), tf.keep(o2), tf.keep(coeffs), mLoss.arraySync(), rLoss.arraySync());
                    
                    return loss;
                });
            });
            
            // Move to the next batch and increment batchIndex
            result = await iterator.next();
            batchIndex++;  // Increment the batch index
            
            await tf.nextFrame();  // Yield control back to the UI for responsiveness
        }
    }
}

export { trainModel, marginLoss, reconstructionLoss };