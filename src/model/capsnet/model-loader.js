import * as tf from "@tensorflow/tfjs";

async function loadModelFromGitHub() {
    const url = "https://raw.githubusercontent.com/zsoltkebel/capsnet-models/main/small/capsnet.json";
    try {
        const capsnet = await tf.loadLayersModel(url);
        const decoder = capsnet.getLayer("Decoder");
        
        console.log("Successfully loaded model from GitHub");
        return [capsnet, decoder];
    } catch (error) {
        console.warn("Could not load model from GitHub: ", error);
    }
}

export { loadModelFromGitHub };