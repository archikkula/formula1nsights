import React, { useState } from 'react';
import './index.css';
import Home from './pages/home';
import Landing from './pages/landing';
import Predictor from './pages/predictor'; // ← Add this import

export default function App() {
  const [currentPage, setCurrentPage] = useState('home');

  const renderPage = () => {
    switch(currentPage) {
      case 'home':
        return (
          <>
            <Landing />
            <Home />
          </>
        );
      case 'predictor':
        return <Predictor />;
      default:
        return (
          <>
            <Landing />
            <Home />
          </>
        );
    }
  };

  return (
    <div className="font-f1-primary bg-f1-black text-f1-white min-h-screen">
      <nav className="fixed top-0 left-0 w-full bg-f1-black text-f1-white z-50 shadow-md h-16 flex items-center px-6 border-b border-gray-800">
        <div className="flex items-center space-x-6">
          <button 
            onClick={() => setCurrentPage('home')}
            className={`text-lg font-f1-primary transition-colors hover:text-f1-red ${
              currentPage === 'home' ? 'text-f1-red border-b-2 border-f1-red pb-1' : 'text-f1-white'
            }`}
          >
            Home
          </button>
          <button 
            onClick={() => setCurrentPage('predictor')}
            className={`text-lg font-f1-primary transition-colors hover:text-f1-red ${
              currentPage === 'predictor' ? 'text-f1-red border-b-2 border-f1-red pb-1' : 'text-f1-white'
            }`}
          >
            Predictor
          </button>
        </div>
        
        <div className="ml-auto">
          <span className="text-f1-red font-f1-bold text-xl">Formula 1nsights</span>
        </div>
      </nav>

      <main className="pt-16 min-h-screen">
        {renderPage()}
      </main>

      <footer className="mt-12 font-f1-bold bg-f1-red text-f1-white px-4 py-2 text-center">
        © {new Date().getFullYear()} Formula 1nsights
      </footer>
    </div>
  );
}