// src/components/UI.jsx
// Composants UI partagés du dashboard FDQN-TE+
// Regroupe : Title, KPI, Panel, DropZone, Badge, PDRTooltip

import { useState, useRef } from "react";
import { C } from "../styles/colors";

/* ─── Title ──────────────────────────────────────────────────────────────── */
export function Title({ children, accent = C.cyan }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 12 }}>
      <div style={{
        width: 3, height: 15, background: accent, borderRadius: 2,
        boxShadow: `0 0 6px ${accent}`
      }} />
      <span style={{
        color: C.txt, fontSize: 11, letterSpacing: 2,
        textTransform: "uppercase", fontFamily: "'Space Mono', monospace"
      }}>
        {children}
      </span>
    </div>
  );
}

/* ─── KPI card ───────────────────────────────────────────────────────────── */
export function KPI({ label, value, unit = "", color = C.cyan, sub = "" }) {
  return (
    <div style={{
      background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8,
      padding: "12px 15px", position: "relative", overflow: "hidden",
      boxShadow: `0 0 16px ${color}12`
    }}>
      <div style={{
        borderTop: `2px solid ${color}`,
        position: "absolute", top: 0, left: 0, right: 0
      }} />
      <div style={{
        color: C.dim, fontSize: 9, letterSpacing: 2,
        textTransform: "uppercase", marginBottom: 4
      }}>
        {label}
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 4 }}>
        <span style={{
          color, fontSize: 22, fontFamily: "'Space Mono', monospace",
          fontWeight: 700, lineHeight: 1
        }}>
          {value}
        </span>
        {unit && <span style={{ color: C.dim, fontSize: 10 }}>{unit}</span>}
      </div>
      {sub && <div style={{ color: C.dim, fontSize: 9, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

/* ─── Panel ──────────────────────────────────────────────────────────────── */
export function Panel({ children, style = {} }) {
  return (
    <div style={{
      background: C.panel,
      border: `1px solid ${C.border}`,
      borderRadius: 8,
      padding: 14,
      ...style,
    }}>
      {children}
    </div>
  );
}

/* ─── DropZone ───────────────────────────────────────────────────────────── */
export function DropZone({ label, accept, onLoad, hint, color = C.cyan }) {
  const ref = useRef(null);
  const [dragging, setDragging] = useState(false);

  const read = (f) => {
    if (!f) return;
    const r = new FileReader();
    r.onload = (e) => onLoad(e.target.result, f.name);
    r.readAsText(f);
  };

  return (
    <div
      onDragOver={e => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={e => { e.preventDefault(); setDragging(false); read(e.dataTransfer.files[0]); }}
      onClick={() => ref.current?.click()}
      style={{
        border: `2px dashed ${dragging ? color : C.border}`,
        borderRadius: 7, padding: "13px 16px", cursor: "pointer",
        background: dragging ? `${color}08` : C.panel,
        transition: "all .2s", textAlign: "center",
        boxShadow: dragging ? `0 0 14px ${color}30` : "none"
      }}
    >
      <div style={{ fontSize: 15, marginBottom: 4 }}>📂</div>
      <div style={{ color: C.txt, fontSize: 11, marginBottom: 2 }}>{label}</div>
      <div style={{ color: C.dim, fontSize: 9 }}>{hint}</div>
      <input
        ref={ref} type="file" accept={accept} style={{ display: "none" }}
        onChange={e => read(e.target.files[0])}
      />
    </div>
  );
}

/* ─── Badge (fichier chargé) ─────────────────────────────────────────────── */
export function Badge({ name, color = C.green, onClear }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 7,
      background: `${color}10`, border: `1px solid ${color}`,
      borderRadius: 5, padding: "4px 10px", fontSize: 9,
      width: "100%", boxSizing: "border-box", justifyContent: "space-between"
    }}>
      <span style={{ color, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
        ✓ {name}
      </span>
      <span
        onClick={e => { e.stopPropagation(); onClear(); }}
        style={{ color: C.dim, cursor: "pointer", fontSize: 13, lineHeight: 1, flexShrink: 0 }}
      >
        ×
      </span>
    </div>
  );
}

/* ─── PDRTooltip (tooltip recharts personnalisé) ─────────────────────────── */
export function PDRTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload ?? {};
  return (
    <div style={{
      background: "#060e1a", border: `1px solid ${C.border}`,
      borderRadius: 7, padding: "9px 13px",
      fontFamily: "'JetBrains Mono', monospace", fontSize: 10,
      minWidth: 200, pointerEvents: "none"
    }}>
      <div style={{ color: C.dim, fontSize: 9, marginBottom: 6 }}>t = {label}s</div>
      {payload.map((p, i) => (
        <div key={i} style={{
          display: "flex", justifyContent: "space-between", gap: 14,
          padding: "2px 0", color: C.dim
        }}>
          <span>{p.name}</span>
          <span style={{ color: p.color ?? C.txt, fontWeight: 700 }}>
            {typeof p.value === "number" ? `${p.value.toFixed(2)}%` : p.value}
          </span>
        </div>
      ))}
      {d.alive != null && (
        <div style={{ borderTop: `1px solid ${C.border}`, marginTop: 5, paddingTop: 5 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 14, fontSize: 9, color: C.dim }}>
            <span>Vivants</span>
            <span style={{ color: C.green }}>{d.alive}</span>
          </div>
          {d.atRisk > 0 && (
            <div style={{ display: "flex", justifyContent: "space-between", gap: 14, fontSize: 9, color: C.dim }}>
              <span>PEPM@risque</span>
              <span style={{ color: C.red }}>{d.atRisk}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
