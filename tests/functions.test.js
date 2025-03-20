import * as tf from '@tensorflow/tfjs';
import { marginLoss, reconstructionLoss } from '../src/model/capsnet/trainer';
import { squash, dynamicRouting } from '../src/model/capsnet/capsnet-tensorflow';

const testCapsuleOutputs = tf.tensor([[
    [ 0.3, 0.4 ],  // caps 0, norm: 0.5
    [0.36, 0.48],  // caps 1, norm: 0.6
    [ 0.6, 0.8 ],  // caps 2, norm: 1
    [0.42, 0.56],  // caps 3, norm: 0.7
]]);

function allClose(tensor1, tensor2, tolerance = 1e-5) {
    // Check if the tensors have the same shape
    if (!tensor1.shape.every((v, i) => v === tensor2.shape[i])) {
        return false;
    }

    // Compute element-wise absolute difference
    const diff = tensor1.sub(tensor2).abs();

    // Check if the difference is within tolerance
    return diff.lessEqual(tf.scalar(tolerance)).all().arraySync();
}

describe("Margin loss function", () => {
    it("returns a small number when correct capsule has the highest activation", async () => {
        const trueLabel = tf.tensor([[0, 0, 1, 0]]);  // one-hot 2

        const loss = marginLoss(trueLabel, testCapsuleOutputs).arraySync();

        expect(loss).toBeCloseTo(0.385);
    });

    it("returns a large number when correct capsule does not have the highest activation", async () => {
        const testTrueLabel = tf.tensor([[1, 0, 0, 0]]); // one-hot 0

        const loss = marginLoss(testTrueLabel, testCapsuleOutputs).arraySync();

        expect(loss).toBeCloseTo(0.87);
    });
});

describe("Reconstruction loss function", () => {
    const image = tf.tensor([[0, 0.1, 0.4, 0.2, 1, 0.2]]);

    it("returns 0 if reconstruction is identical to image", () => {
        const reconstruction = tf.tensor([[0, 0.1, 0.4, 0.2, 1, 0.2]]);

        const loss = reconstructionLoss(image, reconstruction).arraySync();
        
        expect(loss).toBe(0);
    });

    it("calculates loss correctly", () => {
        const reconstruction = tf.tensor([[0, 0.2, 0.3, 0, 0, 0.2]]);

        const loss = reconstructionLoss(image, reconstruction).arraySync();
        
        expect(loss).toBeCloseTo(0.17667);
    });
});

describe("Squash function", () => {
    it("shrinks vectors appropriately", () => {
        const squashed = squash(testCapsuleOutputs);
        const expected = tf.tensor([[
            [0.12, 0.16],
            [0.15882, 0.21176],
            [0.3, 0.4],
            [0.19732, 0.26309],
        ]]);
        squashed.print();
        const res = allClose(squashed, expected);
        console.log(res);
        expect(res).toBeTruthy();
    });
});

describe("Dynamic routing algorithm", () => {
    // for a setup with 1 -> 2 capsules
    // i    j layer
    // o -> o
    //   \
    //    > o
    const u_hat = tf.tensor([[
        [[0.5, 0.2]],  // u_hat 0 given 0
        [[0.3, 0.1]],  // u_hat 1 given 0
    ]]);

    it("works with 1 iteration", () => {
        const expected = squash(tf.tensor([[
            [0.25, 0.1],
            [0.15, 0.05],
        ]]));

        const [v, _] = dynamicRouting(u_hat, 1);

        expect(allClose(v, expected)).toBeTruthy();
    });
});