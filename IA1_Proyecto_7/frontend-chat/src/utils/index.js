import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';
import * as readline from 'readline';

// Cargar los archivos JSON
const modelJson = JSON.parse(fs.readFileSync('modeloo.json'));
const tokenizerJson = JSON.parse(fs.readFileSync('tokenizer.json'));
const intents = JSON.parse(fs.readFileSync('intents.json')).intents;

const wordIndex = JSON.parse(tokenizerJson['config']['word_index']);

// Cargar el modelo desde la memoria
const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson));
console.log(model.summary());

// Función para convertir el texto a secuencias
function textsToSequences(texts) {
    return texts.map(text => {
        const words = text.toLowerCase().trim().split(" ");
        return words.map(word => wordIndex[word] || 0); // Mapea cada palabra al índice correspondiente
    });
}

// Función para rellenar secuencias
function padSequences(sequences, maxLength, paddingType = 'pre', truncatingType = 'pre', paddingValue = 0) {
    return sequences.map(seq => {
        if (seq.length > maxLength) {
            if (truncatingType === 'pre') {
                seq = seq.slice(seq.length - maxLength);
            } else {
                seq = seq.slice(0, maxLength);
            }
        }

        if (seq.length < maxLength) {
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

// Función para obtener la respuesta del chatbot basado en la entrada del usuario
export function getResponse(userInput) {
    userInput = userInput.toLowerCase(); // Convertir a minúsculas

    for (const intent of intents) {
        for (const pattern of intent.patterns) {
            if (userInput.includes(pattern.toLowerCase())) {
                const randomIndex = Math.floor(Math.random() * intent.responses.length);
                return intent.responses[randomIndex];
            }
        }
    }

    return "I'm sorry, I didn't understand that.";
}

// Crear la interfaz de entrada del usuario
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

/*
const main = async () => {
    let user_input = await askQuestion("Ingrese una frase: ");
    user_input = user_input.toLowerCase();

    let sequences = textsToSequences([user_input]);
    console.log("Tokenized sequences:", sequences);

    sequences = padSequences(sequences, 5, 'pre', 'pre', 0);
    console.log("Padded sequences:", sequences);

    // Convertir las secuencias a un tensor
    const tensorInput = tf.tensor2d(sequences);
    console.log("Tensor input:", tensorInput);

    // Hacer una predicción
    const prediction = model.predict(tensorInput);
    prediction.print();  // Imprimir el resultado de la predicción

    // Obtener la respuesta del chatbot
    const response = getResponse(user_input);
    console.log("Respuesta del bot:", response);
}

main();

*/

