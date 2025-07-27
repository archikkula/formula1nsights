// frontend/src/App.js

import React from 'react';
import './index.css';
import Home from './pages/home';
import Landing from './pages/landing';

export default function App() {
  return (
    <div className="font-f1-primary bg-f1-black text-f1-white min-h-screen">
      <nav className="fixed top-0 left-0 w-full bg-f1-black text-f1-white z-50 shadow-md h-16 flex items-center px-6">
        <a href="/"          className="text-f1-white hover:text-f1-red mr-6">Home</a>
        <a href="/drivers"   className="text-f1-white hover:text-f1-red mr-6">Drivers</a>
        <a href="/predictor" className="text-f1-white hover:text-f1-red">Predictor</a>
      </nav>

      <main className="pt-16">
        <Landing />
        
      </main>

      <main className="pt-16">
        <Home />
        
      </main>
      

      <footer className="mt-12 font-f1-bold bg-f1-red text-f1-white px-4 py-2 rounded text-center">
        © {new Date().getFullYear()} Formula 1nsights
      </footer>
    </div>
  );
}
