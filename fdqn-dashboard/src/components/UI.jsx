import { useState, useRef } from "react";
import { C } from "../styles/colors";

/* ─── Composants UI ────────────────────────────────────────────────────────── */
export const Title = ({ children, accent = C.cyan }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
    <div style={{
      width: 3,
      height: 14,
      background: accent,
      borderRadius: 2,
      boxShadow: `0 0 8px ${accent}80`
    }} />
    <span style={{
      color: C.txt,
      fontSize: 10,
      letterSpacing: 2,
      textTransform: "uppercase",
      fontFamily: "'JetBrains Mono', monospace"
    }}>
      {children}
    </span>
  </div>
);

export const KPI = ({ label, value, unit = "", color = C.cyan, sub = "" }) => (
  <div style={{
    background: C.panel,
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    padding: "11px 14px",
    position: "relative",
    overflow: "hidden",
    boxShadow: `0 0 20px ${color}08`
  }}>
    <div style={{
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      height: 2,
      background: color,
      boxShadow: `0 0 8px ${color}`
    }} />
    <div style={{
      color: C.dim,
      fontSize: 9,
      letterSpacing: 2,
      textTransform: "uppercase",
      marginBottom: 3
    }}>
      {label}
    </div>
    <div style={{ display: "flex", alignItems: "baseline", gap: 3 }}>
      <span style={{
        color,
        fontSize: 20,
        fontFamily: "'JetBrains Mono', monospace",
        fontWeight: 700,
        lineHeight: 1
      }}>
        {value}
      </span>
      {unit && <span style={{ color: C.dim, fontSize: 9 }}>{unit}</span>}
    </div>
    {sub && <div style={{ color: C.dim, fontSize: 9, marginTop: 2 }}>{sub}</div>}
  </div>
);

export const Panel = ({ children, style = {} }) => (
  <div style={{
    background: C.panel,
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    padding: 14,
    ...style
  }}>
    {children}
  </div>
);

export const DropZone = ({ label, accept, onLoad, hint, color = C.cyan }) => {
  const ref = useRef(null);
  const [drag, setDrag] = useState(false);

  const read = (f) => {
    if (!f) return;
    const r = new FileReader();
    r.onload = (e) => onLoad(e.target.result, f.name);
    r.readAsText(f);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        read(e.dataTransfer.files[0]);
      }}
      onClick={() => ref.current?.click()}
      style={{
        border: `1.5px dashed ${drag ? color : C.border}`,
        borderRadius: 7,
        padding: "12px 16px",
        cursor: "pointer",
        background: drag ? `${color}08` : C.muted,
        transition: "all .2s",
        textAlign: "center",
        boxShadow: drag ? `0 0 16px ${color}30` : "none"
      }}
    >
      <div style={{ fontSize: 14, marginBottom: 3 }}>📂</div>
      <div style={{ color: C.txt, fontSize: 10, marginBottom: 1 }}>{label}</div>
      <div style={{ color: C.dim, fontSize: 9 }}>{hint}</div>
      <input
        ref={ref}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => read(e.target.files[0])}
      />
    </div>
  );
};

export const Badge = ({ name, color, onClear }) => (
  <div style={{
    display: "inline-flex",
    alignItems: "center",
    gap: 6,
    background: `${color}12`,
    border: `1px solid ${color}50`,
    borderRadius: 5,
    padding: "3px 9px",
    fontSize: 9
  }}>
    <span style={{ color }}>✓ {name}</span>
    <span
      onClick={(e) => {
        e.stopPropagation();
        onClear();
      }}
      style={{
        color: C.dim,
        cursor: "pointer",
        fontSize: 12,
        lineHeight: 1
      }}
    >
      ×
    </span>
  </div>
);

export const PDRTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#060e1a",
      border: `1px solid ${C.border}`,
      borderRadius: 6,
      padding: "8px 12px",
      fontSize: 10
    }}>
      <div style={{ color: C.dim, marginBottom: 4 }}>t = {label}s</div>
      {payload.map((p) => (
        <div
          key={p.name}
          style={{
            color: p.color,
            display: "flex",
            justifyContent: "space-between",
            gap: 16
          }}
        >
          <span>{p.name}</span>
          <span style={{ fontWeight: 700 }}>{p.value?.toFixed(2)}%</span>
        </div>
      ))}
    </div>
  );
};