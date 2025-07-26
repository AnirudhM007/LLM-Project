/** @type {import('tailwindcss').Config} */
export const content = [
  "./src/**/*.{js,jsx,ts,tsx}", // Scans all JS/JSX/TS/TSX files in src folder
  "./public/index.html", // Scans your public HTML file
];
export const theme = {
  extend: {
    fontFamily: {
      inter: ['Inter', 'sans-serif'], // Keep this for your Inter font
    },
  },
};
export const plugins = [];