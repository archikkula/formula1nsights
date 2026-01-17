import React, { useState, useEffect } from 'react';

export default function Landing() {
  //todo: put into separate file
  const allDrivers = [
    { name: "MAX VERSTAPPEN", team: "red_bull", number: "#1", image: "verstappen.png" },
    { name: "LEWIS HAMILTON", team: "ferrari", number: "#44", image: "hamilton.png" },
    { name: "LANDO NORRIS", team: "mclaren", number: "#4", image: "norris.png" },
    { name: "GEORGE RUSSELL", team: "mercedes", number: "#63", image: "russell.png" },
    { name: "OSCAR PIASTRI", team: "mclaren", number: "#81", image: "piastri.png" },
    { name: "CARLOS SAINZ JR", team: "williams", number: "#55", image: "sainz.png" },
    { name: "FERNANDO ALONSO", team: "aston_martin", number: "#14", image: "alonso.png" },
    { name: "YUKI TSUNODA", team: "red_bull", number: "#22", image: "tsunoda.png" },
    { name: "PIERRE GASLY", team: "alpine", number: "#10", image: "gasly.png" },
    { name: "ALEXANDER ALBON", team: "williams", number: "#23", image: "albon.png" },
    { name: "NICO HULKENBERG", team: "kick_sauber", number: "#27", image: "hulkenberg.png" },
    { name: "LANCE STROLL", team: "aston_martin", number: "#18", image: "stroll.png" },
    { name: "ESTEBAN OCON", team: "haas", number: "#31", image: "ocon.png" },
    { name: "LIAM LAWSON", team: "rb", number: "#30", image: "lawson.png" },
    { name: "KIMI ANTONELLI", team: "mercedes", number: "#12", image: "antonelli.png" },
    { name: "OLIVER BEARMAN", team: "haas", number: "#50", image: "bearman.png" },
    { name: "GABRIEL BORTOLETO", team: "kick_sauber", number: "#24", image: "bortoleto.png" },
    { name: "FRANCO COLAPINTO", team: "alpine", number: "#43", image: "colapinto.png" },
    { name: "ISACK HADJAR", team: "rb", number: "#21", image: "hadjar.png" },
    { name: "CHARLES LECLERC", team: "ferrari", number: "#16", image: "leclerc.png" } // Leclerc last
  ];

  // animation constants
  const [currentCarIndex, setCurrentCarIndex] = useState(0);
  const [cycleComplete, setCycleComplete] = useState(false);
  const [carDrivingOff, setCarDrivingOff] = useState(false);
  const [showText, setShowText] = useState(false);
  const [visibleLetters, setVisibleLetters] = useState(0);

  // team colors codes
  const teamColors = {
    'mclaren': '#FF8000',
    'ferrari': '#E8002D', 
    'mercedes': '#27F4D2',
    'red_bull': '#3671C6',
    'williams': '#64C4FF',
    'kick_sauber': '#52E252',
    'rb': '#6692FF',
    'aston_martin': '#229971',
    'haas': '#B6BABD',
    'alpine': '#FF87BC'
  };

  const text = "Formula 1nsights";
  const letters = text.split('');

  //cycle through 20 cars
  useEffect(() => {
    if (!cycleComplete) {
      const interval = setInterval(() => {
        setCurrentCarIndex(prev => {
          const nextIndex = (prev + 1) % allDrivers.length;
          //check if car cycle is over
          if (nextIndex === 0 && prev === allDrivers.length - 1) {
            setCycleComplete(true);
            // delays and then car drives off
            setTimeout(() => {
              setCarDrivingOff(true);
              //start text animation
              setShowText(true);
            }, 300); // 0.3 second pause on last car
          }
          
          return nextIndex;
        });
      }, 100);
      return () => clearInterval(interval);
    }
  }, [cycleComplete, allDrivers.length]);

  //show logo text
  useEffect(() => {
    if (showText && visibleLetters < letters.length) {
      const timer = setTimeout(() => {
        setVisibleLetters(prev => prev + 1);
      }, 40); 
      return () => clearTimeout(timer);
    }
  }, [showText, visibleLetters, letters.length]);

  return (
    <div className="relative flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-4 overflow-hidden">
      {/* flashing car and drive off*/}
      <div className="absolute z-50" style={{ left: '50%', top: '50%' }}>
        <div
          className={`transition-all duration-1000 ease-out ${
            carDrivingOff ? 'transform translate-x-[150vw]' : 'transform translate-x-0'
          }`}
          style={{
            transform: carDrivingOff 
              ? 'translate(-50%, -50%) translateX(150vw)' 
              : 'translate(-50%, -50%)',
          }}
        >
          <img 
            src={`/cars/car_images/${allDrivers[currentCarIndex].image}`}
            alt={`${allDrivers[currentCarIndex].name} car`}
            className="w-[1066px] h-auto object-contain transition-opacity duration-75"
            style={{
              filter: `drop-shadow(0 0 15px ${teamColors[allDrivers[currentCarIndex].team]}30)`,
            }}
          />
        </div>
      </div>

      {/* formula 1nsights main logo */}
      <div className="absolute z-30" style={{ left: '50%', top: '50%', transform: 'translate(-50%, -50%)' }}>
        {showText && (
          <div className="flex justify-center">
            {letters.map((letter, index) => {
              return (
                <div key={index} className="relative">
                  <span
                    className="font-f1-display text-6xl md:text-7xl lg:text-8xl font-bold"
                    style={{
                      color: letter === ' ' ? 'transparent' : '#DC0000',
                      textShadow: letter === ' ' ? 'none' : '0 0 20px #DC000060, 0 0 40px #DC000040',
                      minWidth: letter === ' ' ? '0.5em' : 'auto',
                      opacity: index < visibleLetters ? 1 : 0,
                      transition: 'opacity 500ms ease-out',
                      transitionDelay: `${index * 40}ms`,
                      display: 'inline-block',
                      verticalAlign: 'baseline',
                    }}
                  >
                    {letter === ' ' ? '\u00A0' : letter}
                  </span>
                </div>
              );
            })}
          </div>
        )}
        
        {/* subtitle */}
        {showText && visibleLetters >= letters.length && (
          <div className="absolute top-full left-1/2 transform -translate-x-1/2 mt-8 w-full flex justify-center">
            <p className="text-xl md:text-2xl text-f1-white drop-shadow-xl text-shadow-md opacity-0 animate-fade-in text-center">
              Latest F1 News and Race Predictor
            </p>
          </div>
        )}
      </div>

      {/* css for subtitle */}
      <style jsx>{`
        .text-shadow-md {
          text-shadow: 
            0 0 8px rgba(19, 19, 19, 0.7),
            0 0 16px rgba(19, 19, 19, 0.5),
            1px 1px 3px rgba(19, 19, 19, 0.8);
        }
        
        .animate-fade-in {
          animation: fadeIn 1s ease-out forwards;
          animation-delay: 0.5s;
        }
        
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}