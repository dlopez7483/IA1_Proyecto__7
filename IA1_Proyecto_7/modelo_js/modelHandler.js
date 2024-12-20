import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as readline from 'readline';

const modelJson = JSON.parse(fs.readFileSync('modeloo.json'));
const tokenizerJson = JSON.parse(fs.readFileSync('tokenizer.json'));

const wordIndex = JSON.parse(tokenizerJson['config']['word_index']);

// Cargando el modelo desde la memoria
const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson));
console.log(model.summary());

function textsToSequences(texts) {
    return texts.map(text => {
        const words = text.toLowerCase().trim().split(" ");
        return words.map(word => wordIndex[word] || 0); // Mapea cada palabra al índice correspondiente
    });
}

function padSequences(sequences, maxLength, paddingType = 'pre', truncatingType = 'pre', paddingValue = 0) {
    return sequences.map(seq => {
        if (seq.length > maxLength) {
            // Si la secuencia es más larga que maxLength, se trunca
            if (truncatingType === 'pre') {
                seq = seq.slice(seq.length - maxLength);
            } else {
                seq = seq.slice(0, maxLength);
            }
        }

        if (seq.length < maxLength) {
            // Si la secuencia es más corta que maxLength, se rellena
            const paddingLength = maxLength - seq.length;
            const paddingArray = new Array(paddingLength).fill(paddingValue);

            if (paddingType === 'pre') {
                seq = [...paddingArray, ...seq];
            } else {
                seq = [...seq, ...paddingArray];
            }
        }

        return seq;
    });
}

const r1 = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

const askQuestion = (question) => {
    return new Promise((resolve) => {
        r1.question(question, (answer) => {
            resolve(answer);
        });
    });
}

const main = async () => {
    let user_input = await askQuestion("Ingrese una frase: ");
    user_input = user_input.toLowerCase();

    let sequences = textsToSequences([user_input]);
    console.log("Tokenized sequences:", sequences);

    sequences = padSequences(sequences, 5, 'pre', 'pre', 0);
    console.log("Padded sequences:", sequences);

    // Convert sequences to a tensor
    const tensorInput = tf.tensor2d(sequences);
    console.log("Tensor input:", tensorInput);

    // Make a prediction
    const prediction = model.predict(tensorInput);
    prediction.print();  // Print the prediction result
}

main();
