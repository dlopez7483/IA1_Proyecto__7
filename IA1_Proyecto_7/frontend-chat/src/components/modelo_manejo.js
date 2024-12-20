import * as tf from '@tensorflow/tfjs';
import * as fs from 'fs';

// Leer el modelo y tokenizer usando fs
const modelJson = JSON.parse(fs.readFileSync('model.json'));
const tokenizerJson = JSON.parse(fs.readFileSync('tokenizer.json'));

// Obtener el índice de palabras del tokenizer
const wordIndex = tokenizerJson.config.word_index;

// Cargar el modelo en memoria
const model = await tf.loadLayersModel(tf.io.fromMemory(modelJson));
console.log("Modelo cargado correctamente.");

// Función para convertir el texto en secuencias
const textsToSequences = (texts) => {
  return texts.map(text => {
    const words = text.toLowerCase().trim().split(" ");
    return words.map(word => wordIndex[word] || 0); // Mapea la palabra a su índice
  });
};

// Función para realizar padding de las secuencias
const padSequences = (sequences, maxLength, paddingType = 'pre', truncatingType = 'pre', paddingValue = 0) => {
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
};

// Función para obtener la respuesta del modelo
const getResponse = async (inputText) => {
  const processedText = inputText.toLowerCase().replace(/[^\w\s]/g, "");

  let sequences = textsToSequences([processedText]);
  sequences = padSequences(sequences, 5);

  const tensorInput = tf.tensor2d(sequences);
  const prediction = model.predict(tensorInput);

  const predictedIndex = prediction.argMax(-1).dataSync()[0];
  return `Predicción para el índice: ${predictedIndex}`;
};
