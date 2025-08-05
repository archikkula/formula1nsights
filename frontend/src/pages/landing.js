import React from 'react';

export default function Landing() {
  // Team colors mapping
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

  // Drivers ordered by popularity (most popular first)
  const drivers = [
    { name: "MAX VERSTAPPEN", team: "red_bull", number: "#1" },
    { name: "LEWIS HAMILTON", team: "ferrari", number: "#44" },
    { name: "LANDO NORRIS", team: "mclaren", number: "#4" },
    { name: "CHARLES LECLERC", team: "ferrari", number: "#16" },
    { name: "GEORGE RUSSELL", team: "mercedes", number: "#63" },
    { name: "OSCAR PIASTRI", team: "mclaren", number: "#81" },
    { name: "CARLOS SAINZ JR", team: "williams", number: "#55" },
    { name: "FERNANDO ALONSO", team: "aston_martin", number: "#14" },
    { name: "YUKI TSUNODA", team: "red_bull", number: "#22" },
    { name: "PIERRE GASLY", team: "alpine", number: "#10" },
    { name: "ALEXANDER ALBON", team: "williams", number: "#23" },
    { name: "NICO HULKENBERG", team: "kick_sauber", number: "#27" },
    { name: "LANCE STROLL", team: "aston_martin", number: "#18" },
    { name: "ESTEBAN OCON", team: "haas", number: "#31" },
    { name: "LIAM LAWSON", team: "rb", number: "#30" },
    { name: "KIMI ANTONELLI", team: "mercedes", number: "#12" },
    { name: "OLIVER BEARMAN", team: "haas", number: "#50" },
    { name: "GABRIEL BORTOLETO", team: "kick_sauber", number: "#24" },
    { name: "FRANCO COLAPINTO", team: "alpine", number: "#43" },
    { name: "ISACK HADJAR", team: "rb", number: "#21" }
  ];

  return (
    <div className="relative flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-4 overflow-hidden">
      {/* Animated driver names background */}
      <div className="absolute inset-0 z-0 opacity-15">
        {drivers.map((driver, index) => (
          <div
            key={driver.name}
            className="absolute whitespace-nowrap text-2xl md:text-3xl lg:text-4xl font-f1-display font-black flex"
            style={{
              color: teamColors[driver.team],
              top: `${(index * 4.5) % 90 + 5}%`,
              animation: `scroll-right-${index % 3} ${18 + (index % 3) * 3}s linear infinite`,
              width: '200vw'
            }}
          >
            {/* Create seamless repeating content */}
            <div className="flex animate-scroll">
              {Array(20).fill(null).map((_, i) => (
                <span key={i} className="mr-8 flex-shrink-0">
                  {driver.name} {driver.number}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Main content */}
      <div className="relative z-10 text-center">
        <h1 className="font-f1-display text-8xl md:text-9xl lg:text-[6rem] font-bold text-f1-red mb-6 drop-shadow-2xl text-shadow-lg">
          Formula 1nsights
        </h1>
        <p className="text-xl md:text-2xl text-f1-white drop-shadow-xl text-shadow-md">
          Latest F1 News and Race Predictor
        </p>
      </div>

      {/* CSS Animation */}
      <style jsx>{`
        @keyframes scroll-right-0 {
          0% {
            transform: translateX(-50%);
          }
          100% {
            transform: translateX(0);
          }
        }
        
        @keyframes scroll-right-1 {
          0% {
            transform: translateX(-50%);
          }
          100% {
            transform: translateX(0);
          }
        }
        
        @keyframes scroll-right-2 {
          0% {
            transform: translateX(-50%);
          }
          100% {
            transform: translateX(0);
          }
        }
        
        .text-shadow-lg {
          text-shadow: 
            0 0 15px rgb(19, 19, 19),
            0 0 20px rgb(19, 19, 19),
            0 0 30px rgb(19, 19, 19),
            2px 2px 4px rgb(19, 19, 19);
        }
        
        .text-shadow-md {
          text-shadow: 
            0 0 8px rgb(19, 19, 19),
            0 0 16px rgb(19, 19, 19),
            1px 1px 3px rgb(19, 19, 19);
        }
      `}</style>
    </div>
  );
}