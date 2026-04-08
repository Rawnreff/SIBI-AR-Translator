const tf = require('@tensorflow/tfjs');
const tfn = require('@tensorflow/tfjs-node');

async function testLoad() {
    console.log("Loading model.json...");
    try {
        const handler = tfn.io.fileSystem('./tfjs_model/model.json');
        const model = await tf.loadLayersModel(handler);
        console.log("Model loaded successfully!");
        model.summary();
    } catch (e) {
        console.error("TF.js Error loading model:");
        console.error(e);
    }
}

testLoad();
