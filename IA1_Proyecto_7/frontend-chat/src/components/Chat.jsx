import React, { useState, useRef, useEffect } from "react";
import Mensaje from "./Mensaje";
import * as tf from "@tensorflow/tfjs"; // Importa TensorFlow.js para manejar el modelo

let model;
let tokenizer;

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const chatEndRef = useRef(null); // Para el scroll automático

  // Cargar el modelo y el tokenizador al iniciar el componente
  useEffect(() => {
    const loadModelAndTokenizer = async () => {
      try {
        console.log("Cargando modelo...");
        model = await tf.loadLayersModel("/modelo/modeloo.json"); // Ruta al modelo
        console.log("Modelo cargado correctamente.");
      } catch (error) {
        console.error("Error al cargar el modelo:", error);
      }

      try {
        console.log("Cargando tokenizer...");
        const tokenizerResponse = await fetch("/modelo/tokenizer.json"); // Ruta al tokenizador
        tokenizer = await tokenizerResponse.json();
        console.log("Tokenizer cargado correctamente.");
        if (tokenizer.word_index) {
          console.log("Ejemplo de word_index:", tokenizer.word_index);
        }
        else {
          console.error("word_index no encontrado en el tokenizer.");
        }

      } catch (error) {
        console.error("Error al cargar el tokenizer:", error);
      }
    };

    loadModelAndTokenizer();
  }, []);

  // Hacer scroll hacia abajo cuando se agregan mensajes
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom(); // Scroll automático
  }, [messages]);

  // Tokenizar la entrada del usuario
  const tokenizeInput = (text) => {
    if (!tokenizer || !tokenizer.word_index) {
      console.error("Tokenizer o word_index no cargado correctamente.");
      return [];
    }

    const words = text.toLowerCase().replace(/[^\w\s]/g, "").split(" ");
    return words
      .map((word) => tokenizer.word_index[word] || 0) // Convierte las palabras en índices (0 si no está en word_index)
      .filter((index) => index > 0); // Filtra palabras no encontradas (si es necesario)
  };

  // Padding de las secuencias
  const padSequences = (sequences, maxLength) => {
    const padded = Array.from({ length: maxLength }, () => 0);
    for (let i = 0; i < Math.min(sequences.length, maxLength); i++) {
      padded[i] = sequences[i];
    }
    return [padded];
  };

  // Obtener respuesta del modelo
  const getResponse = async (inputText) => {
    if (!model) {
      console.error("Modelo no cargado.");
      return "Error al cargar el modelo.";
    }
    if (!tokenizer) {
      console.error("tokenazier no cargado.");
      return "Error al cargar el modelo.";
    }
    



    const sequences = tokenizeInput(inputText); // Tokenizar entrada
    const paddedSequences = padSequences(sequences, model.inputs[0].shape[1]); // Padding
    const prediction = model.predict(tf.tensor2d(paddedSequences)); // Predicción

    // Extraer la respuesta generada (si es texto)
    const response = await prediction.data(); // Obtiene las probabilidades o texto generado
    return response.join(" "); // Ajusta según el formato de tu modelo
  };

  // Manejar envío de mensajes
  const handleSendMessage = async () => {
    if (input.trim() === "") return; // No enviar mensajes vacíos

    const timestamp = new Date().toLocaleString();
    const userMessage = { text: input, sender: "user", timestamp };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      const botResponse = await getResponse(input);
      const botMessage = { text: botResponse, sender: "bot", timestamp };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      const errorMessage = {
        text: "Ocurrió un error al procesar tu mensaje. Intenta nuevamente.",
        sender: "bot",
        timestamp,
      };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
      console.error("Error al predecir:", error);
    }

    setInput(""); // Limpiar el campo de entrada
  };

  // Enviar mensaje al presionar Enter
  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleSendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">Modelo IA Fase 1</div>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <Mensaje
            key={index}
            text={msg.text}
            sender={msg.sender}
            timestamp={msg.timestamp}
          />
        ))}
        <div ref={chatEndRef} /> {/* Bandera para el scroll automático */}
      </div>
      <div className="input-group">
        <input
          type="text"
          placeholder="Escribe un mensaje..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
        />
        <button onClick={handleSendMessage}>Enviar</button>
      </div>
    </div>
  );
};

export default Chat;
