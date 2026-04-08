import { useState, useRef, useCallback, useEffect } from "react";
import { C } from "../styles/colors";
import { clCol, hexA } from "../utils/clusterColors";

export default function TopoCanvas({ nodes, selectedId, onSelect }) {
  const canvasRef = useRef(null);
  const st = useRef({ zoom: 1, panX: 0, panY: 0, drag: false, dsx: 0, dsy: 0, psx: 0, psy: 0 });
  const viewRef = useRef("clusters");
  const flagsRef = useRef({ links: true, labels: true, radio: false });
  const pathRef = useRef([]);
  const pktRef = useRef(null);
  const pktTimer = useRef(null);
  const [view, setView] = useState("clusters");
  const [flags, setFlags] = useState({ links: true, labels: true, radio: false });
  const [tip, setTip] = useState(null);
  const AREA = 1000, SX = 500, SY = 500, RADIO = 150;

  useEffect(() => { viewRef.current = view; }, [view]);
  useEffect(() => { flagsRef.current = flags; }, [flags]);

  const getCanvas = useCallback(() => canvasRef.current, []);

  const Sv = useCallback((wx, wy, cv) => {
    const { zoom, panX, panY } = st.current;
    const pad = 40;
    const sc = Math.min((cv.width - pad * 2) / AREA, (cv.height - pad * 2) / AREA) * zoom;
    return {
      sx: cv.width / 2 + panX + (wx - AREA / 2) * sc,
      sy: cv.height / 2 + panY + (wy - AREA / 2) * sc,
      sc
    };
  }, []);

  const Wv = useCallback((sx, sy, cv) => {
    const { zoom, panX, panY } = st.current;
    const pad = 40;
    const sc = Math.min((cv.width - pad * 2) / AREA, (cv.height - pad * 2) / AREA) * zoom;
    return {
      wx: (sx - cv.width / 2 - panX) / sc + AREA / 2,
      wy: (sy - cv.height / 2 - panY) / sc + AREA / 2
    };
  }, []);

  const draw = useCallback(() => {
    const cv = getCanvas();
    if (!cv || !cv.width) return;
    const ctx = cv.getContext("2d");
    const vw = viewRef.current;
    const fl = flagsRef.current;

    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.fillStyle = "#030810";
    ctx.fillRect(0, 0, cv.width, cv.height);

    // Grille
    ctx.strokeStyle = "rgba(13,31,53,.8)";
    ctx.lineWidth = 0.5;
    for (let g = 0; g <= AREA; g += 100) {
      const { sx } = Sv(g, 0, cv);
      ctx.beginPath();
      ctx.moveTo(sx, 0);
      ctx.lineTo(sx, cv.height);
      ctx.stroke();
      const { sy } = Sv(0, g, cv);
      ctx.beginPath();
      ctx.moveTo(0, sy);
      ctx.lineTo(cv.width, sy);
      ctx.stroke();
    }

    // Régions cluster
    if (vw === "clusters") {
      nodes.filter(n => n.isCH && n.isAlive).forEach(ch => {
        const { sx, sy, sc } = Sv(ch.x, ch.y, cv);
        const r = RADIO * sc * 0.8;
        const col = clCol(ch.clusterId);
        const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, r);
        g.addColorStop(0, hexA(col, 0.1));
        g.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      });
    }

    // Portée radio
    if (fl.radio && selectedId !== null) {
      const sel = nodes.find(n => n.id === selectedId);
      if (sel) {
        const { sx, sy, sc } = Sv(sel.x, sel.y, cv);
        const r = RADIO * sc;
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(0,212,255,.35)";
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // Liens
    if (fl.links) {
      nodes.forEach(n => {
        if (!n.isAlive || n.isCH) return;
        const ch = nodes.find(c => c.isCH && c.clusterId === n.clusterId && c.isAlive);
        if (!ch) return;
        const { sx: x1, sy: y1 } = Sv(n.x, n.y, cv);
        const { sx: x2, sy: y2 } = Sv(ch.x, ch.y, cv);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.strokeStyle = hexA(clCol(n.clusterId), 0.18);
        ctx.lineWidth = 0.6;
        ctx.stroke();
      });
    }

    // Chemin sélectionné
    if (pathRef.current.length > 1) {
      ctx.beginPath();
      pathRef.current.forEach((n, i) => {
        const { sx, sy } = Sv(n.x, n.y, cv);
        i === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      });
      const { sx: ex, sy: ey } = Sv(SX, SY, cv);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = "rgba(0,255,136,.6)";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Nœuds
    nodes.forEach(n => {
      const { sx, sy } = Sv(n.x, n.y, cv);
      if (sx < -20 || sx > cv.width + 20 || sy < -20 || sy > cv.height + 20) return;
      let col, r;
      if (!n.isAlive) {
        col = "#160505";
        r = 2;
      } else if (n.isCH) {
        col = clCol(n.clusterId);
        r = Math.max(5, 7 * Math.min(st.current.zoom, 2));
        const g = ctx.createRadialGradient(sx, sy, 0, sx, sy, r + 8);
        g.addColorStop(0, hexA(col, 0.5));
        g.addColorStop(1, "transparent");
        ctx.beginPath();
        ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      } else {
        r = Math.max(2.5, 3.5 * Math.min(st.current.zoom, 1.5));
        const en = n.energyNorm ?? 0;
        if (vw === "energy") col = en > 0.7 ? "#00ff88" : en > 0.4 ? "#ff8c00" : "#ff3355";
        else if (vw === "risk") col = n.pepmRisk > 0.7 ? "#ff3355" : n.pepmRisk > 0.4 ? "#ff8c00" : "#264a70";
        else col = hexA(clCol(n.clusterId), 0.8);
      }
      if (n.id === selectedId) {
        ctx.beginPath();
        ctx.arc(sx, sy, r + 8, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(255,255,255,.6)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = col;
      ctx.fill();
      if (n.isCH && fl.labels && r > 4) {
        ctx.fillStyle = "rgba(255,255,255,.85)";
        ctx.font = `bold ${Math.max(8, Math.round(9 * Math.min(st.current.zoom, 1.3)))}px monospace`;
        ctx.textAlign = "center";
        ctx.fillText("#" + n.id, sx, sy - r - 4);
      }
    });

    // Sink
    const { sx: ssx, sy: ssy } = Sv(SX, SY, cv);
    const rs = Math.max(8, 10 * Math.min(st.current.zoom, 1.5));
    [2.5, 1.8, 1.2].forEach((_, i) => {
      ctx.beginPath();
      ctx.arc(ssx, ssy, rs * (2.5 - i * 0.5), 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255,215,0,${0.06 + i * 0.05})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
    ctx.beginPath();
    ctx.arc(ssx, ssy, rs, 0, Math.PI * 2);
    ctx.fillStyle = "#ffd700";
    ctx.fill();
    ctx.fillStyle = "rgba(0,0,0,.8)";
    ctx.font = `bold ${Math.round(rs * 0.85)}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("BS", ssx, ssy);
    ctx.textBaseline = "alphabetic";

    // Paquet animé
    if (pktRef.current) {
      const { sx: px, sy: py } = Sv(pktRef.current.x, pktRef.current.y, cv);
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#00ff88";
      ctx.fill();
    }
  }, [nodes, selectedId, Sv, getCanvas]);

  const animPath = useCallback((srcId) => {
    if (pktTimer.current) clearInterval(pktTimer.current);
    pktRef.current = null;
    const src = nodes.find(n => n.id === srcId);
    if (!src) return { ids: [], delivered: false };
    const path = [src];
    const vis = new Set([src.id]);
    let cur = src;
    while (path.length < 25) {
      let best = null;
      let bd = Math.hypot(cur.x - SX, cur.y - SY);
      nodes.forEach(nb => {
        if (vis.has(nb.id) || !nb.isAlive) return;
        if (Math.hypot(cur.x - nb.x, cur.y - nb.y) > RADIO) return;
        const d = Math.hypot(nb.x - SX, nb.y - SY);
        if (d < bd) {
          bd = d;
          best = nb;
        }
      });
      if (!best) break;
      path.push(best);
      vis.add(best.id);
      cur = best;
      if (bd < 60) break;
    }
    pathRef.current = path;
    const fin = path[path.length - 1];
    const delivered = fin && Math.hypot(fin.x - SX, fin.y - SY) < RADIO;
    const full = [...path, { x: SX, y: SY, id: -1 }];
    let t = 0;
    pktTimer.current = setInterval(() => {
      t++;
      const sf = t / 20;
      const si = Math.floor(sf);
      const sp = sf - si;
      if (si >= full.length - 1) {
        clearInterval(pktTimer.current);
        pktRef.current = null;
        draw();
        return;
      }
      const a = full[si];
      const b = full[si + 1];
      pktRef.current = { x: a.x + (b.x - a.x) * sp, y: a.y + (b.y - a.y) * sp };
      draw();
    }, 35);
    return { ids: path.map(n => n.id), delivered };
  }, [nodes, draw]);

  const onMM = useCallback((e) => {
    const cv = getCanvas();
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const s = st.current;
    if (s.drag) {
      s.panX = s.psx + (mx - s.dsx);
      s.panY = s.psy + (my - s.dsy);
      draw();
      return;
    }
    const { wx, wy } = Wv(mx, my, cv);
    let hov = null;
    let bd = 15 / s.zoom;
    nodes.forEach(n => {
      const d = Math.hypot(n.x - wx, n.y - wy);
      if (d < bd) {
        bd = d;
        hov = n;
      }
    });
    if (hov) {
      const nb = nodes.filter(n => n.isAlive && n.id !== hov.id && Math.hypot(n.x - hov.x, n.y - hov.y) <= RADIO).length;
      setTip({ x: e.clientX, y: e.clientY, node: { ...hov, nb } });
    } else setTip(null);
  }, [nodes, draw, Wv, getCanvas]);

  const onMD = useCallback((e) => {
    const cv = getCanvas();
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const s = st.current;
    s.drag = true;
    s.dsx = e.clientX - rect.left;
    s.dsy = e.clientY - rect.top;
    s.psx = s.panX;
    s.psy = s.panY;
  }, [getCanvas]);

  const onMU = useCallback((e) => {
    const cv = getCanvas();
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const s = st.current;
    if (Math.abs(mx - s.dsx) < 5 && Math.abs(my - s.dsy) < 5) {
      const { wx, wy } = Wv(mx, my, cv);
      let best = null;
      let bd = 18 / s.zoom;
      nodes.forEach(n => {
        const d = Math.hypot(n.x - wx, n.y - wy);
        if (d < bd) {
          bd = d;
          best = n;
        }
      });
      if (best) {
        const r = animPath(best.id);
        const nb = nodes.filter(n => n.isAlive && n.id !== best.id && Math.hypot(n.x - best.x, n.y - best.y) <= RADIO).length;
        onSelect({ ...best, nb }, r);
      }
    }
    s.drag = false;
  }, [nodes, onSelect, animPath, Wv, getCanvas]);

  const onWh = useCallback((e) => {
    e.preventDefault();
    st.current.zoom = Math.max(0.2, Math.min(12, st.current.zoom * (e.deltaY < 0 ? 1.15 : 0.87)));
    draw();
  }, [draw]);

  // Initialisation du canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const resizeObserver = new ResizeObserver(() => {
      const container = canvas.parentElement;
      if (container) {
        canvas.width = container.clientWidth;
        canvas.height = Math.max(350, container.clientWidth * 0.55);
        draw();
      }
    });

    const container = canvas.parentElement;
    if (container) {
      resizeObserver.observe(container);
      canvas.width = container.clientWidth;
      canvas.height = Math.max(350, container.clientWidth * 0.55);
      draw();
    }

    return () => resizeObserver.disconnect();
  }, [draw]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ display: "flex", gap: 5, marginBottom: 6, flexWrap: "wrap", alignItems: "center" }}>
        <select
          value={view}
          onChange={e => setView(e.target.value)}
          style={{
            background: C.muted,
            border: `1px solid ${C.border}`,
            color: C.txt,
            borderRadius: 5,
            padding: "3px 7px",
            fontSize: 9,
            fontFamily: "'JetBrains Mono', monospace"
          }}
        >
          {[["clusters", "Clusters IFO"], ["energy", "Énergie"], ["risk", "Risque PEPM"]].map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </select>
        {[["links", "Liens"], ["labels", "Labels CH"], ["radio", "Portée"]].map(([k, l]) => (
          <button
            key={k}
            onClick={() => setFlags(f => ({ ...f, [k]: !f[k] }))}
            style={{
              background: flags[k] ? `${C.cyan}15` : C.muted,
              border: `1px solid ${flags[k] ? C.cyan : C.border}`,
              color: flags[k] ? C.cyan : C.dim,
              borderRadius: 5,
              padding: "3px 8px",
              fontSize: 9,
              cursor: "pointer",
              fontFamily: "'JetBrains Mono', monospace"
            }}
          >
            {l}
          </button>
        ))}
        <button
          onClick={() => {
            st.current.zoom = 1;
            st.current.panX = 0;
            st.current.panY = 0;
            draw();
          }}
          style={{
            background: C.muted,
            border: `1px solid ${C.border}`,
            color: C.dim,
            borderRadius: 5,
            padding: "3px 8px",
            fontSize: 9,
            cursor: "pointer"
          }}
        >
          ⌖ Reset
        </button>
      </div>
      <div style={{ flex: 1, position: "relative", minHeight: 350 }}>
        <canvas
          ref={canvasRef}
          style={{
            display: "block",
            width: "100%",
            cursor: "crosshair",
            borderRadius: 6,
            border: `1px solid ${C.border}`
          }}
          onMouseMove={onMM}
          onMouseDown={onMD}
          onMouseUp={onMU}
          onMouseLeave={() => {
            setTip(null);
            st.current.drag = false;
          }}
          onWheel={onWh}
        />
        {tip && (
          <div style={{
            position: "fixed",
            left: tip.x + 15,
            top: tip.y - 8,
            background: "#060d1a",
            border: `1px solid ${C.cyan}`,
            borderRadius: 7,
            padding: "9px 13px",
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 10,
            zIndex: 9999,
            minWidth: 190,
            pointerEvents: "none"
          }}>
            <div style={{ color: C.cyan, fontWeight: 700, marginBottom: 5 }}>
              #{tip.node.id}{tip.node.isCH ? " ⬡ CH" : ""}
            </div>
            {[
              ["Énergie", `${(tip.node.energy ?? 0).toFixed(4)} J`],
              ["PEPM risk", `${((tip.node.pepmRisk ?? 0) * 100).toFixed(1)}%`],
              ["Voisins", `${tip.node.nb ?? 0}`],
              ["Dist. Sink", `${(tip.node.distToSink ?? 0).toFixed(1)} m`]
            ].map(([k, v]) => (
              <div key={k} style={{ display: "flex", justifyContent: "space-between", gap: 14, padding: "2px 0", color: C.dim }}>
                <span>{k}</span>
                <span style={{ color: C.txt }}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}