/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        'f1-primary': ['Rajdhani', 'sans-serif'],
        'f1-display': ['Orbitron', 'sans-serif'],
        'f1-bold': ['Michroma', 'sans-serif'],
      },
      colors: {
        'f1-red': '#DC0000',
        'f1-black': '#1A1A1A',
        'f1-white': '#FFFFFF',
      },
    },
  },
  plugins: [],
}