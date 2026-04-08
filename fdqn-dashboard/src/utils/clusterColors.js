import { C } from "../styles/colors";

/* ─── Couleurs cluster ─────────────────────────────────────────────────────── */
export const CPAL = [
  "#00d4ff", "#00ff88", "#ff6b6b", "#ffd700", "#ff8c00", "#a78bfa", "#06b6d4",
  "#84cc16", "#f43f5e", "#8b5cf6", "#0ea5e9", "#22c55e", "#ec4899", "#f97316",
  "#14b8a6", "#eab308", "#6366f1", "#ef4444", "#10b981", "#3b82f6", "#d946ef",
  "#f59e0b", "#64748b", "#fb923c", "#e879f9", "#0891b2", "#16a34a", "#dc2626",
  "#ca8a04", "#7c3aed"
];

export const clCol = (id) => CPAL[Math.abs((typeof id === "number" ? id : parseInt(id) || 0) % CPAL.length)];

export const hexA = (hex, a) => {
  if (!hex?.startsWith("#")) return hex;
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
};

export { C };