/* ─── Palette de couleurs ─────────────────────────────────────────────────── */
export const C = {
  bg: "#04080f",
  panel: "#080f1c",
  border: "#0d1f35",
  cyan: "#00e5ff",
  green: "#00ff9f",
  amber: "#ffaa00",
  red: "#ff3355",
  purple: "#b46eff",
  blue: "#3b82f6",
  txt: "#c8dff0",
  dim: "#3a5570",
  muted: "#0f1e30",
};

export const tt = {
  contentStyle: {
    background: "#060e1a",
    border: `1px solid ${C.border}`,
    borderRadius: 6,
    fontSize: 10,
    color: C.txt,
    padding: "8px 12px"
  },
  labelStyle: { color: C.dim, fontSize: 9 },
  cursor: { stroke: C.border },
};