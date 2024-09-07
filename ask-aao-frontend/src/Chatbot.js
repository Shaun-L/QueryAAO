// src/Chatbot.js
import React, { useState } from 'react';
import axios from 'axios';

const Chatbot = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');

  const handleQueryChange = (event) => {
    setQuery(event.target.value);
  };

  const sendQuery = async () => {
    try {
      const res = await axios.post('http://127.0.0.1:5000/query', { query });
      setResponse(res.data.response);
    } catch (error) {
      console.error("Error fetching response:", error);
      setResponse("An error occurred. Please try again.");
    }
  };

  return (
    <div>
      <h1>QueryAAO</h1>
      <textarea 
        value={query} 
        onChange={handleQueryChange} 
        rows="4" 
        cols="50" 
        placeholder="Ask your question here..."
      />
      <br />
      <button onClick={sendQuery}>Send Query</button>
      <h3>Response:</h3>
      <p>{response}</p>
    </div>
  );
};

export default Chatbot;
