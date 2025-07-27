import React from 'react';

export default function Landing() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-4rem)] px-4">
      <h1
  className="
    font-f1-display text-8xl md:text-9xl lg:text-[6rem]
    font-bold text-f1-red
    [ -webkit-text-stroke:2px_#fff ]
    [ -webkit-text-fill-color:#DC0000 ]
    mb-6
  "
>
  Formula 1nsights
</h1>

      <p className="text-xl md:text-2xl text-f1-white">
        Latest F1 News and Race Predictor
      </p>
    </div>
  );
}
