import React, { useState, useRef, useEffect } from "react";
import Mensaje from "./Mensaje";
import * as tf from '@tensorflow/tfjs';
import { use } from "react";

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [model, setModel] = useState(null);
  const [tokenizer, setTokenizer] = useState(null);
  const [intents, setIntents] = useState(null);
  const chatEndRef = useRef(null);

  // Función para cargar los archivos JSON (modelo, tokenizer, intents)
  const loadFiles = async () => {
    // Cargar el modelo
   try {
    const modelJson = await fetch('mids/model.json');
    const modelData = await modelJson.json();
    const loadedModel = await tf.loadLayersModel(tf.io.fromMemory(modelData));
    setModel(loadedModel);
    
    // Cargar el tokenizer
    const tokenizerJson = await fetch('mids/tokenizer.json');
    const tokenizerData = await tokenizerJson.json();
    setTokenizer(tokenizerData);
    
    // Cargar los intents
    const intentsJson = await fetch('mids/intents.json');
    const intentsData = await intentsJson.json();
    setIntents(intentsData.intents);
    }
    catch (error) {
      console.log(error);
    }
  };

  useEffect(() => {
    loadFiles(); // Cargar los archivos cuando el componente se monta
  }, []);

  // Función para convertir texto a secuencias
  const textsToSequences = (texts) => {



    console.log(texts);
    try {
      
      if (!tokenizer) return [];
      return texts.map((text) => {
        const words = text.toLowerCase().trim().split(" ");
        console.log("Aqui");
        const word = "are";
        const wordIndex = JSON.parse (tokenizer.config.word_index);
       
        


        return words.map((word) => wordIndex[word] || 0); // Mapea cada palabra al índice correspondiente
      });
    }
    catch (error) {
      console.log(error);
    }
    
  };

  // Función para obtener la respuesta del modelo
  const getResponse = async (userInput) => {

   // console.log("Tokenizer: ", tokenizer);
   // console.log("Tokenizer Word Index: ", tokenizer?.word_index);

    if (!model || !tokenizer || !intents) {
      return "Cargando el modelo, por favor espera...";
    }

   
    // Convertir el texto en secuencias y hacer padding
    const sequences = textsToSequences([userInput]);
    const paddedSequences= padSequences(sequences, model.inputs[0].shape[1]); 
    
    
    // Realizar la predicción
    const prediction = model.predict(tf.tensor2d(paddedSequences));
    const responseIndex = prediction.argMax(-1).dataSync()[0];
    const probabilities = prediction.dataSync();
console.log("Probabilidades predichas:", probabilities);

    console.log("Sequences: ", sequences);
    console.log("Padded Sequences: ", paddedSequences);
    console.log("Prediction: ", prediction);
    

    console.log("Índice de respuesta:", responseIndex);
  
    // Verificar si el índice de respuesta es válido y acceder al intent
    console.log("Intents: ", intents.length);
    if (responseIndex < 0 || responseIndex >= intents.length) {
      console.log("Índice de respuesta fuera de rango:", responseIndex);
      return "Lo siento, no pude entender tu solicitudddddd. ¿Puedes intentar otra vez?";
    }
  
    // Obtener el intent correspondiente usando el índice
    const intent = intents[responseIndex];
  
    console.log("Intent encontrado:", intent);
  
    // Seleccionar una respuesta aleatoria del intent
    const randomIndex = Math.floor(Math.random() * intent.responses.length);
    return intent.responses[randomIndex];
  };

  // Padding de las secuencias
  const padSequences = (sequences, maxLength) => {
    return sequences.map((seq) => {
      const padded = Array.from({ length: maxLength }, () => 0);
      for (let i = 0; i < Math.min(seq.length, maxLength); i++) {
        padded[i] = seq[i];
      }
      return padded;
    });
  };

  // Manejo del envío de mensajes
  const handleSendMessage = async () => {
    console.log("Input: ", input);
    if (input.trim() === "") return;

    const timestamp = new Date().toLocaleString();
    const userMessage = { text: input, sender: "user", timestamp };
    setMessages((prevMessages) => [...prevMessages, userMessage]);

    try {
      console.log(input);
      const botResponse = await getResponse(input);
      const botMessage = { text: botResponse, sender: "bot", timestamp };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.log(error);
      const errorMessage = { text: "Ocurrió un error. Envía un mensaje nuevamente...", sender: "bot", timestamp };
      setMessages((prevMessages) => [...prevMessages, errorMessage]);
    }

    setInput("");
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleSendMessage();
    }
  };

  // Función para hacer scroll automáticamente cuando se agregan nuevos mensajes
  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="chat-container">
      <div className="chat-header">Modelo IA Fase 1</div>
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <Mensaje key={index} text={msg.text} sender={msg.sender} timestamp={msg.timestamp} />
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
