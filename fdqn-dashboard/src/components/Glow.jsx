// src/components/Glow.jsx
export const Glow = ({ color = "#00e5ff", intensity = 0.3 }) => (
  <div style={{
    position: "absolute", 
    inset: 0, 
    pointerEvents: "none",
    background: `radial-gradient(ellipse at 50% 0%, ${color}${Math.round(intensity * 255).toString(16).padStart(2,'0')} 0%, transparent 65%)`,
  }} />
);