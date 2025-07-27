// frontend/src/index.js
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';  // ← This must come before anything else

import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
