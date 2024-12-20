import React from "react";

const Mensaje = ({ text, sender, timestamp }) => {
    const isUser = sender === "user";
    return (
      <div className={`chat-message ${isUser ? "user" : "bot"}`}>
        <div className="message-bubble">
          <p>{text}</p>
          <span className="message-timestamp">{sender + ': '}{timestamp}</span>
        </div>
      </div>
    );
};

export default Mensaje;

/*import React from 'react';

const Mensaje = ({ message }) => {
  const isBot = message.sender === 'bot';
  return (
    <div className={`message ${isBot ? 'bot' : 'user'}`}>
      <p>{message.text}</p>
    </div>
  );
};

export default Mensaje;*/