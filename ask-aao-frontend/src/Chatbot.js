import React, { useState } from 'react';
import axios from 'axios';

const Chatbot = () => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [setupMessage, setSetupMessage] = useState('');

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

  const handleSetup = async () => {
    try {
      const res = await axios.post('http://127.0.0.1:5000/setup');
      setSetupMessage(res.data.message);
    } catch (error) {
      console.error("Error during setup:", error);
      setSetupMessage("An error occurred during setup. Please try again.");
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
      <button onClick={handleSetup}>Setup Application</button> {/* New setup button */}
      <h3>Response:</h3>
      <p>{response}</p>
      <h3>Setup Message:</h3>
      <p>{setupMessage}</p> {/* Display setup messages */}
    </div>
  );
};

export default Chatbot;
