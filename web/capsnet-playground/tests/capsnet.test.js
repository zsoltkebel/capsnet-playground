import { squash } from "../src/capsnet.js";
import * as tf from '@tensorflow/tfjs';

describe("squash function", () => {
    test("2D capsules", async () => {
        const tensor = tf.tensor([
            [3, 4],
            [2, 0],
            [1, 1]
        ]);
        const output = squash(tensor);
        output.print();

        const expectedResult = tf.tensor([
            [0.5769231, 0.7692308],
            [0.8, 0],
            [0.4714045, 0.4714045]
        ]);

        const result = Boolean(await tf.equal(output, expectedResult).all().array());
        console.log(result)
        expect(result).toBe(true);
    });

    test("4D capsules", async () => {
        const tensor = tf.tensor(
            [1, 2, 2, 4]
        );
        const expectedResult = tf.tensor(
            [0.1923077, 0.3846154, 0.3846154, 0.7692308]
        )
        const output = squash(tensor);
        output.print();

        const result = Boolean(await tf.equal(output, expectedResult).all().array());
        expect(result).toBe(true);
    })
});