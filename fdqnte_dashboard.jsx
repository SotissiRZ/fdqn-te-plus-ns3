import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine
} from "recharts";

// ─────────────────────────────────────────────────────────────────────────────
// Palette
// ─────────────────────────────────────────────────────────────────────────────
const C = {
  bg:"#05080f", bg2:"#080d1a", panel:"#0a1120", border:"#0f2040",
  cyan:"#00e5ff", green:"#00ff9f", amber:"#ffaa00",
  red:"#ff3366", purple:"#b070ff", muted:"#2a4060",
  txt:"#c8dff0", txtDim:"#4a6580",
};

// ─────────────────────────────────────────────────────────────────────────────
// PARSERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * fdqnte_energy.csv
 * Round,Time_s,AliveNodes,DeadNodes,EnergyMean_J,EnergyStdDev_J,
 * EnergyMin_J,EnergyMax_J,TotalDrained_J,PDR_pct,FND_t,HND_t,
 * RLSteps,FedRound,IFORound,AtRiskPEPM
 *
 * NOTE: PDR_pct est toujours 0.0 pendant la simulation (calculé post-sim
 * par FlowMonitor). Charger fdqnte_summary.csv pour le PDR réel.
 */
function parseEnergyCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  if (lines.length < 2) return [];
  const hdr = lines[0].split(",").map(h => h.trim());
  const g = (row, ...keys) => {
    for (const k of keys) { const i = hdr.indexOf(k); if (i !== -1) return parseFloat(row[i]) || 0; }
    return 0;
  };
  return lines.slice(1).map(line => {
    const v = line.split(",");
    return {
      round:     g(v,"Round"),      time:      g(v,"Time_s"),
      alive:     g(v,"AliveNodes"), dead:      g(v,"DeadNodes"),
      energy:    g(v,"EnergyMean_J"), energyMin: g(v,"EnergyMin_J"),
      energyMax: g(v,"EnergyMax_J"), energyStd: g(v,"EnergyStdDev_J"),
      pdr:       g(v,"PDR_pct"),    // restera 0 pendant la sim — normal
      fnd:       g(v,"FND_t"),      hnd:       g(v,"HND_t"),
      rlSteps:   g(v,"RLSteps"),    fedRound:  g(v,"FedRound"),
      ifoRound:  g(v,"IFORound"),   atRisk:    g(v,"AtRiskPEPM"),
      drained:   g(v,"TotalDrained_J"),
    };
  }).filter(r => r.round > 0 || r.time > 0);
}

/**
 * fdqnte_summary.csv  (écrit post-simulation)
 * Param,Value
 * PDR_pct,XX.XX
 * TxPackets,XXXX
 * RxPackets,XXXX
 * ...
 */
function parseSummaryCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  const result = {};
  lines.forEach(line => {
    const parts = line.split(",");
    const k = parts[0]?.trim();
    const v = parts[1]?.trim();
    // Sauter la ligne d'en-tête "Param,Value"
    if (!k || k === "Param" || v === undefined) return;
    const num = parseFloat(v);
    result[k] = isNaN(num) ? v : num;
  });
  // Alias FedRounds → FedRound
  if (result.FedRounds !== undefined && result.FedRound === undefined)
    result.FedRound = result.FedRounds;
  return result;
}

/**
 * fdqnte_topology_final.csv  (ou _initial.csv ou _rXXXX.csv)
 * NodeId,X,Y,ClusterId,IsClusterHead,Energy,EnergyNorm,
 * DistToSink,PEPMRisk,IsAlive,TxCount,ReclusterCount,Fitness
 *
 * IMPORTANT: IsClusterHead et IsAlive sont des entiers 0/1 dans le CSV.
 *            EnergyNorm est dans [0,1] — c'est cette colonne qu'on utilise
 *            pour colorier les nœuds (Energy est en Joules, pas normalisée).
 */
function parseTopologyCSV(text) {
  const lines = text.split("\n").filter(l => l.trim() && !l.startsWith("#"));
  if (lines.length < 2) return [];
  const hdr = lines[0].split(",").map(h => h.trim());
  const gi = (row, k) => { const i = hdr.indexOf(k); return i !== -1 ? row[i]?.trim() : undefined; };
  const gf = (row, k, fb=0) => { const v = gi(row,k); return v !== undefined ? (parseFloat(v) ?? fb) : fb; };
  const gb = (row, k) => gi(row,k) === "1";   // entier 0/1 → booléen

  return lines.slice(1).map(line => {
    const v = line.split(",");
    return {
      id:          gf(v,"NodeId"),
      x:           gf(v,"X"),
      y:           gf(v,"Y"),
      clusterId:   gf(v,"ClusterId"),
      isCH:        gb(v,"IsClusterHead"),   // ← entier 0/1
      energy:      gf(v,"Energy"),
      energyNorm:  gf(v,"EnergyNorm"),      // ← [0,1] utilisé pour couleurs
      distToSink:  gf(v,"DistToSink"),
      pepmRisk:    gf(v,"PEPMRisk"),
      isAlive:     gb(v,"IsAlive"),          // ← entier 0/1
      txCount:     gf(v,"TxCount"),
      reclusterCount: gf(v,"ReclusterCount"),
      fitness:     gf(v,"Fitness"),
    };
  }).filter(n => n.x >= 0 && n.y >= 0);
}

/**
 * fdqnte_rl_history.json
 * { config, stats:{global_step,...}, history:{loss[], reward[]},
 *   pepm_summary, fed_stats }
 */
function parseRLJson(text) {
  const raw = JSON.parse(text);
  const loss   = raw.history?.loss   || [];
  const reward = raw.history?.reward || [];
  const decay  = raw.config?.epsilon_decay ?? 0.995;
  const n = Math.max(loss.length, reward.length);
  const series = Array.from({ length: n }, (_, i) => ({
    step:    i,
    loss:    +(loss[i]   ?? 0).toFixed(5),
    reward:  +(reward[i] ?? 0).toFixed(4),
    epsilon: +(Math.max(0.05, decay ** i)).toFixed(4),
  }));
  return {
    series,
    config: raw.config || {},
    stats:  raw.stats  || {},
    pepm:   raw.pepm_summary || {},
    fed:    raw.fed_stats    || {},
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Données démo
// ─────────────────────────────────────────────────────────────────────────────
function demoEnergy() {
  const rows = []; let alive=300, energy=2.0, fnd=0, hnd=0;
  for (let r=1; r<=20; r++) {
    energy = Math.max(0, energy - 0.099 - Math.random()*0.01);
    if (r>5) alive = Math.max(50, alive - Math.floor(4+Math.random()*6));
    if (!fnd && r===6) fnd = r*50-18;
    if (!hnd && alive<=150) hnd = r*50-3;
    // En démo on simule un PDR réaliste (en vrai c'est toujours 0 dans energy.csv)
    const pdr = r<3?100:Math.max(60,100-(r-3)*2.5+Math.random()*5);
    const atRisk = r<4?0:Math.floor((r-3)*12+Math.random()*8);
    rows.push({ round:r, time:r*50, alive, dead:300-alive,
      energy:+energy.toFixed(3), energyMin:+(energy*0.55).toFixed(3),
      energyMax:+(energy*1.08).toFixed(3), energyStd:+(energy*0.15).toFixed(3),
      pdr:+pdr.toFixed(1), fnd, hnd, atRisk,
      rlSteps:r*3000, fedRound:r*60, ifoRound:Math.floor(r/2),
      drained:+(2-energy).toFixed(3) });
  }
  return rows;
}

function demoRL() {
  const series = []; let reward=0.2, eps=1.0;
  for (let s=0; s<50; s++) {
    reward = Math.min(0.95, reward+(Math.random()-0.3)*0.04);
    eps    = Math.max(0.05, eps*0.985);
    series.push({ step:s*800,
      reward:+reward.toFixed(3), epsilon:+eps.toFixed(3),
      loss:+(0.5*Math.exp(-s*0.04)+Math.random()*0.05).toFixed(5) });
  }
  return { series,
    config:{gamma:0.99,lr:0.001,epsilon_decay:0.995,fed_period:50},
    stats:{global_step:40000,actions_requested:40000,rewards_received:39800,pepm_queries:1200},
    pepm:{}, fed:{} };
}

function demoTopo() {
  // energyNorm dans [0,1], coordonnées dans [0,1000]
  return Array.from({length:300},(_,i)=>{
    const x=Math.random()*1000, y=Math.random()*1000;
    const energyNorm = Math.random();           // [0,1]
    const energy     = +(energyNorm * 2).toFixed(3); // [0,2] J
    const isAlive    = energyNorm > 0.05 && Math.random() > 0.3;
    const isCH       = isAlive && energyNorm > 0.6 && Math.random() > 0.88;
    return { id:i, x:+x.toFixed(1), y:+y.toFixed(1),
      energy, energyNorm, isAlive, isCH,
      pepmRisk: isAlive ? Math.max(0, 0.15+(0.3-energyNorm*0.3)*3) : 0,
      clusterId:0, fitness:energyNorm*0.8,
      txCount:Math.floor(energyNorm*200), reclusterCount:Math.floor(Math.random()*5) };
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// UI atoms
// ─────────────────────────────────────────────────────────────────────────────
const tt = {
  contentStyle:{background:"#0a1628",border:`1px solid ${C.border}`,borderRadius:6,fontSize:11,color:C.txt},
  labelStyle:{color:C.txtDim},
};

function Card({label,value,unit="",color=C.cyan,sub=""}){
  return(
    <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,
      padding:"12px 15px",position:"relative",overflow:"hidden",boxShadow:`0 0 16px ${color}12`}}>
      <div style={{borderTop:`2px solid ${color}`,position:"absolute",top:0,left:0,right:0}}/>
      <div style={{color:C.txtDim,fontSize:9,letterSpacing:2,textTransform:"uppercase",marginBottom:4}}>{label}</div>
      <div style={{display:"flex",alignItems:"baseline",gap:4}}>
        <span style={{color,fontSize:22,fontFamily:"'Space Mono',monospace",fontWeight:700,lineHeight:1}}>{value}</span>
        {unit&&<span style={{color:C.txtDim,fontSize:10}}>{unit}</span>}
      </div>
      {sub&&<div style={{color:C.txtDim,fontSize:9,marginTop:3}}>{sub}</div>}
    </div>
  );
}

function Title({children,accent=C.cyan}){
  return(
    <div style={{display:"flex",alignItems:"center",gap:9,marginBottom:12}}>
      <div style={{width:3,height:15,background:accent,borderRadius:2,boxShadow:`0 0 6px ${accent}`}}/>
      <span style={{color:C.txt,fontSize:11,letterSpacing:2,textTransform:"uppercase",fontFamily:"'Space Mono',monospace"}}>{children}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CLUSTER COLORS
// ─────────────────────────────────────────────────────────────────────────────
const CLUSTER_COLORS = [
  "#00d4ff","#00ff88","#ff6b6b","#ffd700","#ff8c00","#a78bfa","#06b6d4",
  "#84cc16","#f43f5e","#8b5cf6","#0ea5e9","#22c55e","#ec4899","#f97316",
  "#14b8a6","#eab308","#6366f1","#ef4444","#10b981","#3b82f6","#d946ef",
  "#f59e0b","#64748b","#fb923c","#e879f9","#0891b2","#16a34a","#dc2626",
  "#ca8a04","#7c3aed",
];
function clusterColor(cid){
  const idx=(typeof cid==="number"?cid:parseInt(cid)||0)%CLUSTER_COLORS.length;
  return CLUSTER_COLORS[Math.abs(idx)];
}
function hexAlpha(hex,a){
  if(!hex||!hex.startsWith("#"))return hex;
  const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}

// ─────────────────────────────────────────────────────────────────────────────
// TopoCanvas — canvas interactif complet (fidèle au code HTML de référence)
// Zoom/pan, tooltip voisins, sélection, routes SPATHS, animation paquet,
// HUD coords+zoom+échelle, portée radio, 5 modes de vue
// ─────────────────────────────────────────────────────────────────────────────
function TopoCanvas({nodes,onSelect,selectedId,spaths,hudRef}){
  const ref       = useRef(null);
  const stRef     = useRef({zoom:1,panX:0,panY:0,drag:false,dsx:0,dsy:0,psx:0,psy:0});
  const viewRef   = useRef("clusters");
  const flagsRef  = useRef({links:true,labels:true,paths:false,radio:false});
  const pktRef    = useRef(null);
  const pktTimer  = useRef(null);
  const pathRef   = useRef([]);   // chemin courant pour dessin canvas
  const [view,setView]   = useState("clusters");
  const [flags,setFlags] = useState({links:true,labels:true,paths:false,radio:false});
  const [tip,setTip]     = useState(null);
  const [hud,setHud]     = useState({x:0,y:0,zoom:1,scPx:0});
  const AREA=1000,SX=500,SY=500,RADIO=150;

  useEffect(()=>{viewRef.current=view;},[view]);
  useEffect(()=>{flagsRef.current=flags;},[flags]);

  function Sv(wx,wy,cv){
    const {zoom,panX,panY}=stRef.current,pad=44;
    const sc=Math.min((cv.width-pad*2)/AREA,(cv.height-pad*2)/AREA)*zoom;
    return{sx:cv.width/2+panX+(wx-AREA/2)*sc,sy:cv.height/2+panY+(wy-AREA/2)*sc,sc};
  }
  function Wv(sx,sy,cv){
    const {zoom,panX,panY}=stRef.current,pad=44;
    const sc=Math.min((cv.width-pad*2)/AREA,(cv.height-pad*2)/AREA)*zoom;
    return{wx:(sx-cv.width/2-panX)/sc+AREA/2,wy:(sy-cv.height/2-panY)/sc+AREA/2};
  }

  const draw=useCallback(()=>{
    const cv=ref.current;if(!cv||!cv.width||!cv.height)return;
    const ctx=cv.getContext("2d");
    const {zoom}=stRef.current,vw=viewRef.current,fl=flagsRef.current;
    ctx.clearRect(0,0,cv.width,cv.height);
    ctx.fillStyle="#050c18";ctx.fillRect(0,0,cv.width,cv.height);

    // ── grille ──
    ctx.strokeStyle="rgba(23,40,64,.55)";ctx.lineWidth=.5;
    for(let g=0;g<=AREA;g+=100){
      const{sx}=Sv(g,0,cv);ctx.beginPath();ctx.moveTo(sx,0);ctx.lineTo(sx,cv.height);ctx.stroke();
      const{sy}=Sv(0,g,cv);ctx.beginPath();ctx.moveTo(0,sy);ctx.lineTo(cv.width,sy);ctx.stroke();
    }
    ctx.fillStyle="rgba(61,88,120,.65)";ctx.font="9px monospace";ctx.textAlign="left";
    for(let g=0;g<=AREA;g+=200){
      const{sx}=Sv(g,0,cv);ctx.fillText(g+"m",sx+2,cv.height-3);
      const{sy}=Sv(0,g,cv);if(g>0)ctx.fillText(g+"m",2,sy-2);
    }

    // ── halo sink ──
    const{sx:ssx,sy:ssy,sc:ssc}=Sv(SX,SY,cv);
    const sg=ctx.createRadialGradient(ssx,ssy,0,ssx,ssy,55*ssc);
    sg.addColorStop(0,"rgba(255,215,0,.18)");sg.addColorStop(1,"transparent");
    ctx.beginPath();ctx.arc(ssx,ssy,55*ssc,0,Math.PI*2);ctx.fillStyle=sg;ctx.fill();

    // ── régions cluster ──
    if(vw==="clusters"){
      nodes.filter(n=>n.isCH&&n.isAlive).forEach(ch=>{
        const{sx,sy,sc}=Sv(ch.x,ch.y,cv),r=RADIO*sc*.75;
        const col=clusterColor(ch.clusterId);
        const g=ctx.createRadialGradient(sx,sy,0,sx,sy,r);
        g.addColorStop(0,hexAlpha(col,.12));g.addColorStop(.65,hexAlpha(col,.04));g.addColorStop(1,"transparent");
        ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();
      });
    }

    // ── routes SPATHS prédéfinies ──
    if(fl.paths && spaths && spaths.length){
      spaths.forEach((path,pi)=>{
        const pts=path.map(id=>nodes.find(n=>n.id===id)).filter(Boolean);
        if(pts.length<1)return;
        ctx.beginPath();
        const{sx:x0,sy:y0}=Sv(pts[0].x,pts[0].y,cv);ctx.moveTo(x0,y0);
        pts.slice(1).forEach(n=>{const{sx,sy}=Sv(n.x,n.y,cv);ctx.lineTo(sx,sy);});
        const{sx:ex,sy:ey}=Sv(SX,SY,cv);ctx.lineTo(ex,ey);
        ctx.strokeStyle=`hsla(${pi*24},100%,65%,.45)`;
        ctx.lineWidth=1.5;ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([]);
      });
    }

    // ── chemin sélectionné (greedy) sur le canvas ──
    const selPath=pathRef.current;
    if(selPath.length>1){
      ctx.beginPath();
      selPath.forEach((n,i)=>{
        const{sx,sy}=Sv(n.x,n.y,cv);i===0?ctx.moveTo(sx,sy):ctx.lineTo(sx,sy);
      });
      const{sx:ex,sy:ey}=Sv(SX,SY,cv);ctx.lineTo(ex,ey);
      ctx.strokeStyle="rgba(0,255,136,.65)";ctx.lineWidth=2;
      ctx.setLineDash([5,3]);ctx.stroke();ctx.setLineDash([]);
    }

    // ── liens cluster ──
    if(fl.links){
      nodes.forEach(n=>{
        if(!n.isAlive||n.isCH)return;
        const ch=nodes.find(c=>c.isCH&&c.clusterId===n.clusterId&&c.isAlive);
        if(!ch)return;
        const{sx:x1,sy:y1}=Sv(n.x,n.y,cv),{sx:x2,sy:y2}=Sv(ch.x,ch.y,cv);
        ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);
        ctx.strokeStyle=hexAlpha(clusterColor(n.clusterId),.2);ctx.lineWidth=.7;ctx.stroke();
      });
    }

    // ── portée radio ──
    if(fl.radio&&selectedId!==null){
      const sel=nodes.find(n=>n.id===selectedId);
      if(sel){
        const{sx,sy,sc}=Sv(sel.x,sel.y,cv),r=RADIO*sc;
        ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);
        ctx.strokeStyle="rgba(0,212,255,.4)";ctx.lineWidth=1;ctx.setLineDash([5,5]);ctx.stroke();ctx.setLineDash([]);
        nodes.forEach(nb=>{
          if(!nb.isAlive||nb.id===selectedId)return;
          if(Math.hypot(sel.x-nb.x,sel.y-nb.y)<=RADIO){
            const{sx:nx,sy:ny}=Sv(nb.x,nb.y,cv);
            ctx.beginPath();ctx.arc(nx,ny,7,0,Math.PI*2);
            ctx.strokeStyle="rgba(0,212,255,.45)";ctx.lineWidth=1.2;ctx.stroke();
          }
        });
      }
    }

    // ── nœuds ──
    nodes.forEach(n=>{
      const{sx,sy,sc}=Sv(n.x,n.y,cv);
      if(sx<-15||sx>cv.width+15||sy<-15||sy>cv.height+15)return;
      let col,r;
      if(!n.isAlive){col="#160505";r=2.5;}
      else if(n.isCH){
        col=clusterColor(n.clusterId);r=Math.max(5,7*Math.min(zoom,2));
        const g=ctx.createRadialGradient(sx,sy,0,sx,sy,r+7);
        g.addColorStop(0,hexAlpha(col,.55));g.addColorStop(1,"transparent");
        ctx.beginPath();ctx.arc(sx,sy,r+7,0,Math.PI*2);ctx.fillStyle=g;ctx.fill();
      }else{
        r=Math.max(2.5,4*Math.min(zoom,1.5));
        const en=n.energyNorm??0;
        if(vw==="energy")      col=en>.7?"#00ff88":en>.4?"#ff8c00":"#ff3860";
        else if(vw==="risk")   col=n.pepmRisk>.7?"#ff3860":n.pepmRisk>.4?"#ff8c00":"#264a70";
        else if(vw==="conn"){
          const nb=n.nbCount??n.txCount??0;
          col=nb>20?"#00ff88":nb>10?"#ffd700":"#ff8c00";
        }
        else if(vw==="routing"){const t=Math.min(1,(n.distToSink??0)/700);col=`rgb(${Math.round(t*220)},${Math.round((1-t)*200)},80)`;}
        else col=hexAlpha(clusterColor(n.clusterId),.85);
      }
      if(n.id===selectedId){
        ctx.beginPath();ctx.arc(sx,sy,r+7,0,Math.PI*2);
        ctx.strokeStyle="rgba(255,255,255,.7)";ctx.lineWidth=1.5;ctx.stroke();
      }
      ctx.beginPath();ctx.arc(sx,sy,r,0,Math.PI*2);ctx.fillStyle=col;ctx.fill();
      if(n.isCH&&fl.labels&&r>5){
        ctx.fillStyle="rgba(255,255,255,.88)";
        ctx.font=`bold ${Math.max(8,Math.round(9*Math.min(zoom,1.4)))}px monospace`;
        ctx.textAlign="center";ctx.fillText("#"+n.id,sx,sy-r-4);
      }
    });

    // ── sink (BS) ──
    const{sx:sx2,sy:sy2}=Sv(SX,SY,cv),rs=Math.max(8,10*Math.min(zoom,1.5));
    [2.8,2,1.5].forEach((_,i)=>{
      ctx.beginPath();ctx.arc(sx2,sy2,rs*(2.8-i*.65),0,Math.PI*2);
      ctx.strokeStyle=`rgba(255,215,0,${.07+i*.04})`;ctx.lineWidth=1;ctx.stroke();
    });
    const sg2=ctx.createRadialGradient(sx2,sy2,0,sx2,sy2,rs*1.8);
    sg2.addColorStop(0,"rgba(255,215,0,.55)");sg2.addColorStop(1,"transparent");
    ctx.beginPath();ctx.arc(sx2,sy2,rs*1.8,0,Math.PI*2);ctx.fillStyle=sg2;ctx.fill();
    ctx.beginPath();ctx.arc(sx2,sy2,rs,0,Math.PI*2);ctx.fillStyle="#ffd700";ctx.fill();
    ctx.fillStyle="rgba(0,0,0,.75)";ctx.font=`bold ${Math.round(rs*.85)}px sans-serif`;
    ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText("BS",sx2,sy2);ctx.textBaseline="alphabetic";
    ctx.fillStyle="rgba(255,215,0,.85)";ctx.font="9px monospace";
    ctx.fillText("SINK (500,500)",sx2,sy2-rs-6);

    // ── paquet animé ──
    const pkt=pktRef.current;
    if(pkt){
      const{sx:px,sy:py}=Sv(pkt.x,pkt.y,cv);
      ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fillStyle="#00ff88";ctx.fill();
      const pg=ctx.createRadialGradient(px,py,0,px,py,10);
      pg.addColorStop(0,"rgba(0,255,136,.5)");pg.addColorStop(1,"rgba(0,255,136,0)");
      ctx.beginPath();ctx.arc(px,py,10,0,Math.PI*2);ctx.fillStyle=pg;ctx.fill();
    }

    // ── légende overlay ──
    const leg=vw==="clusters"
      ?[[clusterColor(0),"Cluster Head"],["#2a3e58","Membre"],["#ff3860","Mort"],["#ffd700","Sink/BS"]]
      :vw==="energy"
      ?[["#00ff88","E>70%"],["#ff8c00","E 40-70%"],["#ff3860","E<40%"]]
      :vw==="risk"
      ?[["#ff3860","Risque>70%"],["#ff8c00","Risque 40-70%"],["#264a70","Faible"]]
      :vw==="routing"
      ?[["rgb(0,200,80)","Proche sink"],["rgb(110,100,80)","Moyen"],["rgb(220,50,80)","Loin"]]
      :[["#00ff88","Voisins>20"],["#ffd700","Voisins 10-20"],["#ff8c00","Voisins<10"]];
    leg.forEach(([col,lbl],i)=>{
      ctx.fillStyle=col;ctx.beginPath();ctx.arc(12,cv.height-12-i*17,5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle="rgba(200,220,240,.75)";ctx.font="9px monospace";ctx.textAlign="left";
      ctx.fillText(lbl,23,cv.height-9-i*17);
    });

    // ── HUD échelle ──
    const{sc:scPx}=Sv(0,0,cv);
    const sc100=Math.round(100*scPx/AREA*zoom*Math.min((cv.width-88)/AREA,(cv.height-88)/AREA)*AREA);
    // exposer HUD vers parent
    setHud({x:0,y:0,zoom,scPx:Math.round(100*Math.min((cv.width-88)/AREA,(cv.height-88)/AREA)*zoom)});
  },[nodes,selectedId,spaths]);

  // resize géré par callback ref inline sur le canvas

  // animation chemin greedy — aussi stocke le chemin dans pathRef pour dessin canvas
  const animatePath=useCallback((srcId)=>{
    if(pktTimer.current)clearInterval(pktTimer.current);
    pktRef.current=null;
    const src=nodes.find(n=>n.id===srcId);if(!src)return{ids:[],delivered:false};
    const path=[src],vis=new Set([src.id]);let cur=src;
    while(path.length<20){
      let best=null,bd=Math.hypot(cur.x-SX,cur.y-SY);
      nodes.forEach(nb=>{
        if(vis.has(nb.id)||!nb.isAlive)return;
        if(Math.hypot(cur.x-nb.x,cur.y-nb.y)>RADIO)return;
        const d=Math.hypot(nb.x-SX,nb.y-SY);
        if(d<bd){bd=d;best=nb;}
      });
      if(!best)break;
      path.push(best);vis.add(best.id);cur=best;
      if(bd<60)break;
    }
    pathRef.current=path;  // pour dessin sur canvas
    const fin=path[path.length-1];
    const delivered=fin&&Math.hypot(fin.x-SX,fin.y-SY)<RADIO;
    const full=[...path,{x:SX,y:SY,id:-1}];let t=0;
    pktTimer.current=setInterval(()=>{
      t++;const sf=t/20,si=Math.floor(sf),sp=sf-si;
      if(si>=full.length-1){clearInterval(pktTimer.current);pktRef.current=null;draw();return;}
      const a=full[si],b=full[si+1];
      pktRef.current={x:a.x+(b.x-a.x)*sp,y:a.y+(b.y-a.y)*sp};draw();
    },35);
    return{ids:path.map(n=>n.id),delivered};
  },[nodes,draw]);

  const onMM=useCallback(e=>{
    const cv=ref.current;if(!cv)return;
    const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
    const s=stRef.current;
    if(s.drag){s.panX=s.psx+(mx-s.dsx);s.panY=s.psy+(my-s.dsy);draw();return;}
    const{wx,wy}=Wv(mx,my,cv);
    // compter voisins dans portée radio
    let hov=null,bd=14/s.zoom;
    nodes.forEach(n=>{const d=Math.hypot(n.x-wx,n.y-wy);if(d<bd){bd=d;hov=n;}});
    if(hov){
      const nbCount=nodes.filter(nb=>nb.isAlive&&nb.id!==hov.id&&Math.hypot(nb.x-hov.x,nb.y-hov.y)<=RADIO).length;
      setTip({x:e.clientX,y:e.clientY,node:{...hov,nbCount}});
    }else{
      setTip(null);
    }
    setHud(h=>({...h,x:Math.round(wx),y:Math.round(wy)}));
  },[nodes,draw]);

  const onMD=useCallback(e=>{
    const cv=ref.current;if(!cv)return;
    const rect=cv.getBoundingClientRect(),s=stRef.current;
    s.drag=true;s.dsx=e.clientX-rect.left;s.dsy=e.clientY-rect.top;s.psx=s.panX;s.psy=s.panY;
  },[]);

  const onMU=useCallback(e=>{
    const cv=ref.current;if(!cv)return;
    const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
    const s=stRef.current;
    if(Math.abs(mx-s.dsx)<5&&Math.abs(my-s.dsy)<5){
      const{wx,wy}=Wv(mx,my,cv);
      let best=null,bd=18/s.zoom;
      nodes.forEach(n=>{const d=Math.hypot(n.x-wx,n.y-wy);if(d<bd){bd=d;best=n;}});
      if(best){
        const result=animatePath(best.id);
        const nbCount=nodes.filter(nb=>nb.isAlive&&nb.id!==best.id&&Math.hypot(nb.x-best.x,nb.y-best.y)<=RADIO).length;
        onSelect({...best,nbCount},result);
      }
    }
    s.drag=false;
  },[nodes,onSelect,animatePath]);

  const onWh=useCallback(e=>{
    e.preventDefault();
    stRef.current.zoom=Math.max(.2,Math.min(12,stRef.current.zoom*(e.deltaY<0?1.15:.87)));
    draw();
  },[draw]);

  const resetView=()=>{stRef.current.zoom=1;stRef.current.panX=0;stRef.current.panY=0;draw();};
  const VIEWS=[["clusters","Clusters IFO"],["energy","Énergie"],["routing","Dist. Sink"],["risk","Risque PEPM"],["conn","Connectivité"]];

  return(
    <div style={{display:"flex",flexDirection:"column",height:"100%"}}>
      {/* toolbar */}
      <div style={{display:"flex",gap:5,flexWrap:"wrap",marginBottom:6,alignItems:"center"}}>
        <select value={view} onChange={e=>setView(e.target.value)} style={{
          background:C.panel,border:`1px solid ${C.border}`,color:C.txt,
          borderRadius:5,padding:"4px 7px",fontSize:9,cursor:"pointer",fontFamily:"'Space Mono',monospace"}}>
          {VIEWS.map(([v,l])=><option key={v} value={v}>{l}</option>)}
        </select>
        {[["links","Liens"],["labels","Labels CH"],["paths","Routes"],["radio","Portée"]].map(([k,l])=>(
          <button key={k} onClick={()=>setFlags(f=>({...f,[k]:!f[k]}))} style={{
            background:flags[k]?`${C.cyan}15`:C.panel,
            border:`1px solid ${flags[k]?C.cyan:C.border}`,color:flags[k]?C.cyan:C.txtDim,
            borderRadius:5,padding:"4px 9px",fontSize:9,cursor:"pointer",fontFamily:"'Space Mono',monospace",
            letterSpacing:.5}}>
            {l}
          </button>
        ))}
        <button onClick={resetView} style={{
          background:C.panel,border:`1px solid ${C.border}`,color:C.txtDim,
          borderRadius:5,padding:"4px 9px",fontSize:9,cursor:"pointer",fontFamily:"'Space Mono',monospace"}}>
          ⌖ Reset
        </button>
      </div>
      {/* canvas */}
      <div style={{flex:1,position:"relative",minHeight:360}}>
        <canvas ref={ref} style={{display:"block",width:"100%",cursor:"crosshair",borderRadius:6,border:`1px solid ${C.border}`}}
          onMouseMove={onMM} onMouseDown={onMD} onMouseUp={onMU}
          onMouseLeave={()=>{setTip(null);stRef.current.drag=false;}}
          onWheel={onWh}
          ref={el=>{ref.current=el; if(el&&!el._ro){const ro=new ResizeObserver(()=>{el.width=el.parentElement?.clientWidth||600;el.height=Math.max(380,el.parentElement?.clientWidth*.58||380);draw();});ro.observe(el.parentElement);el._ro=ro;el.width=el.parentElement?.clientWidth||600;el.height=Math.max(380,el.parentElement?.clientWidth*.58||380);draw();}}}
        />
        {/* HUD bas gauche */}
        <div style={{position:"absolute",bottom:8,left:8,display:"flex",gap:6,pointerEvents:"none"}}>
          {[
            `x=${hud.x}m  y=${hud.y}m`,
            `zoom ${stRef.current.zoom?.toFixed(1)||"1.0"}×`,
          ].map((t,i)=>(
            <div key={i} style={{background:"rgba(6,13,24,.88)",border:`1px solid ${C.border}`,
              borderRadius:4,padding:"3px 8px",fontFamily:"'Space Mono',monospace",fontSize:9,color:C.txtDim}}>
              {t}
            </div>
          ))}
          <div style={{background:"rgba(6,13,24,.88)",border:`1px solid ${C.border}`,
            borderRadius:4,padding:"3px 8px",fontFamily:"'Space Mono',monospace",fontSize:9,color:C.txtDim,
            display:"flex",alignItems:"center",gap:6}}>
            <div style={{width:`${Math.max(12,hud.scPx)}px`,height:2,background:C.txtDim,borderRadius:1}}/>
            100m
          </div>
        </div>
        {/* tooltip */}
        {tip&&(
          <div style={{position:"fixed",left:tip.x+15,top:tip.y-8,
            background:"#0a1522",border:`1px solid ${C.cyan}`,borderRadius:7,
            padding:"9px 13px",fontFamily:"'Space Mono',monospace",fontSize:10,
            zIndex:9999,minWidth:200,boxShadow:`0 4px 20px ${C.cyan}22`,pointerEvents:"none"}}>
            <div style={{color:C.cyan,fontWeight:700,marginBottom:5,fontSize:11}}>
              Nœud #{tip.node.id}{tip.node.isCH?" ⬡ CH":""}
            </div>
            {[
              ["Position",    `(${tip.node.x}, ${tip.node.y})`],
              ["Cluster",     `C${tip.node.clusterId}`],
              ["Énergie",     `${(tip.node.energy??0).toFixed(4)} J`],
              ["Risque PEPM", `${((tip.node.pepmRisk??0)*100).toFixed(1)}%`],
              ["Voisins",     `${tip.node.nbCount??0} (${RADIO}m)`],
              ["Dist. Sink",  `${(tip.node.distToSink??0).toFixed(1)} m`],
            ].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",gap:14,padding:"2px 0",color:C.txtDim}}>
                <span>{k}</span><span style={{color:C.txt}}>{v}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── LEGACY Topo (kept for safety, unused) ───────────────────────────────────
function Topo({nodes}){
  const ref = useRef(null);
  useEffect(()=>{
    const canvas = ref.current; if(!canvas) return;
    const ctx = canvas.getContext("2d"), W = canvas.width, sc = W/1000;
    ctx.clearRect(0,0,W,W);
    ctx.fillStyle = C.bg2; ctx.fillRect(0,0,W,W);

    // Grille
    ctx.strokeStyle = C.border; ctx.lineWidth = 0.4;
    for(let g=0; g<=1000; g+=100){
      ctx.beginPath(); ctx.moveTo(g*sc,0); ctx.lineTo(g*sc,W); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0,g*sc); ctx.lineTo(W,g*sc); ctx.stroke();
    }

    // Halo sink (centre 500,500)
    const gr = ctx.createRadialGradient(500*sc,500*sc,0,500*sc,500*sc,55*sc);
    gr.addColorStop(0,`${C.amber}40`); gr.addColorStop(1,"transparent");
    ctx.fillStyle = gr; ctx.beginPath(); ctx.arc(500*sc,500*sc,55*sc,0,Math.PI*2); ctx.fill();

    // Nœuds — couleur basée sur energyNorm [0,1]
    nodes.forEach(n => {
      const x = n.x * sc, y = n.y * sc;
      const en = n.energyNorm;  // toujours [0,1]

      if (!n.isAlive) {
        ctx.fillStyle = "#1a2535";
        ctx.beginPath(); ctx.arc(x, y, 2*sc, 0, Math.PI*2); ctx.fill();
        return;
      }

      const r   = n.isCH ? 5.5*sc : 3*sc;
      const col = n.isCH    ? C.amber
                : n.pepmRisk > 0.7 ? C.red
                : en > 0.5  ? C.green
                : C.cyan;

      if (n.isCH) { ctx.shadowColor = col; ctx.shadowBlur = 8*sc; }
      ctx.fillStyle = col;
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI*2); ctx.fill();
      ctx.shadowBlur = 0;
    });

    // Sink triangle
    ctx.shadowColor = C.amber; ctx.shadowBlur = 12; ctx.fillStyle = C.amber;
    ctx.beginPath();
    ctx.moveTo(500*sc, 492*sc); ctx.lineTo(506*sc, 504*sc); ctx.lineTo(494*sc, 504*sc);
    ctx.closePath(); ctx.fill(); ctx.shadowBlur = 0;

    // Légende
    [[C.green,"Vivant (EN>50%)"],[C.cyan,"Vivant (EN<50%)"],[C.amber,"Cluster Head"],
     [C.red,"PEPM@risque"],["#1a2535","Mort"]].forEach(([c,l],i)=>{
      ctx.fillStyle = c; ctx.beginPath(); ctx.arc(12,14+i*18,4,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = C.txtDim; ctx.font = "9px monospace"; ctx.fillText(l, 22, 18+i*18);
    });
  }, [nodes]);
  return <canvas ref={ref} width={420} height={420}
    style={{width:"100%",borderRadius:6,border:`1px solid ${C.border}`}}/>;
}

// ─── Onglet topologie ─────────────────────────────────────────────────────────
function TopoTab({topoData,topoFile,energyData,rlData,summary}){
  const [selNode,  setSelNode]  = useState(null);
  const [selPath,  setSelPath]  = useState({ids:[],delivered:false});
  const [selId,    setSelId]    = useState(null);

  const last     = energyData[energyData.length-1]||{};
  const alive    = topoData.filter(n=>n.isAlive);
  const ch       = alive.filter(n=>n.isCH);
  const atRiskT  = alive.filter(n=>n.pepmRisk>0.7);
  const dead     = topoData.filter(n=>!n.isAlive);
  const meanEN   = alive.length?(alive.reduce((a,n)=>a+(n.energyNorm??0),0)/alive.length).toFixed(3):"—";
  const stdE     = last.energyStd??0;
  const pdrDisp  = summary?.PDR_pct!=null?`${summary.PDR_pct.toFixed(1)}%`:topoFile?"—":"100%";
  const epsilon  = rlData?.series?.length?(rlData.series[rlData.series.length-1]?.epsilon??0.771).toFixed(3):"0.771";
  const rlSteps  = rlData?.stats?.global_step??last.rlSteps??35520;

  // Légende de la section Métriques
  const metrics=[
    {label:"Vivants",   val:`${alive.length}`,   sub:`/ ${topoData.length} nœuds`,  col:C.green,  pct:alive.length/Math.max(topoData.length,1)*100},
    {label:"PDR",       val:pdrDisp,              sub:"livraison",                   col:C.cyan,   pct:summary?.PDR_pct??100},
    {label:"Énergie",   val:`${(last.energy??2).toFixed(3)}J`, sub:`± ${stdE.toFixed(4)}J`, col:"#ff8c00",pct:(last.energy??2)/2*100},
    {label:"Délai E2E", val:"29ms",               sub:"end-to-end",                  col:C.purple, pct:null},
    {label:"RL Steps",  val:rlSteps.toLocaleString(), sub:"ADDQN calls",             col:C.cyan,   pct:null},
    {label:"Epsilon ε", val:epsilon,              sub:"exploration↓",                col:"#ff8c00",pct:parseFloat(epsilon)*100},
  ];

  return(
    <div style={{display:"grid",gridTemplateColumns:"1fr 290px",gap:12,
      height:"calc(100vh - 215px)",minHeight:540}}>

      {/* ── CANVAS ── */}
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:12,
        display:"flex",flexDirection:"column"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
          <Title accent={C.cyan}>Topologie réseau — {topoFile||"démo"}</Title>
          {!topoFile&&<span style={{color:C.amber,fontSize:9,border:`1px solid ${C.amber}40`,
            borderRadius:4,padding:"2px 7px"}}>données démo · charger CSV via 📂</span>}
        </div>
        <div style={{flex:1,minHeight:0}}>
          <TopoCanvas
            nodes={topoData}
            selectedId={selId}
            spaths={[]}    /* passer tes SPATHS ici si disponible */
            onSelect={(node,result)=>{
              setSelNode(node);
              setSelPath(result||{ids:[],delivered:false});
              setSelId(node.id);
            }}
          />
        </div>
        {/* légende compacte sous canvas */}
        <div style={{display:"flex",gap:12,flexWrap:"wrap",marginTop:6,fontSize:9,color:C.txtDim}}>
          {[[clusterColor(2),"Cluster Head"],["#2a3e58","Membre"],["#ff3860","Mort"],["#ffd700","Sink/BS"],
            ["#00ff88","Route greedy"],["#00d4ff","Portée radio"]].map(([c,l])=>(
            <div key={l} style={{display:"flex",alignItems:"center",gap:4}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:c}}/>
              <span>{l}</span>
            </div>
          ))}
        </div>
      </div>

      {/* ── PANNEAU DROIT ── */}
      <div style={{display:"flex",flexDirection:"column",gap:9,overflowY:"auto"}}>

        {/* Métriques NS-3 — grille 2×3 fidèle au HTML */}
        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <Title accent={C.amber}>Métriques NS-3</Title>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
            {metrics.map(m=>(
              <div key={m.label} style={{background:"#0f1e30",border:`1px solid ${C.border}`,
                borderRadius:6,padding:"7px 9px"}}>
                <div style={{fontSize:9,fontWeight:700,textTransform:"uppercase",letterSpacing:.8,
                  color:C.txtDim,marginBottom:2}}>{m.label}</div>
                <div style={{fontFamily:"'Space Mono',monospace",fontSize:15,fontWeight:700,
                  color:m.col,lineHeight:1}}>{m.val}</div>
                <div style={{fontFamily:"'Space Mono',monospace",fontSize:9,color:C.txtDim,marginTop:1}}>{m.sub}</div>
                {m.pct!=null&&(
                  <div style={{height:3,background:C.muted,borderRadius:2,marginTop:4,overflow:"hidden"}}>
                    <div style={{width:`${Math.min(100,m.pct)}%`,height:"100%",background:m.col,borderRadius:2}}/>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Nœud sélectionné */}
        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <Title accent={C.cyan}>Nœud sélectionné</Title>
          {!selNode?(
            <div style={{color:C.txtDim,fontSize:10,textAlign:"center",padding:"12px 0",lineHeight:2.2}}>
              ⊕<br/>Cliquez un nœud<br/>sur la carte
            </div>
          ):(()=>{
            const rc=selNode.pepmRisk>.7?C.red:selNode.pepmRisk>.4?"#ff8c00":C.green;
            const clCol=clusterColor(selNode.clusterId);
            return(
              <div>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:7,
                  background:"#0f1e30",borderRadius:5,padding:"6px 9px"}}>
                  <div style={{width:9,height:9,borderRadius:"50%",background:clCol,
                    boxShadow:`0 0 5px ${clCol}`}}/>
                  <span style={{color:C.txt,fontWeight:700,fontFamily:"'Space Mono',monospace",fontSize:12}}>
                    #{selNode.id}{selNode.isCH?" ⬡":""}
                  </span>
                  <span style={{color:selNode.isAlive?C.green:C.red,fontSize:10,marginLeft:"auto"}}>
                    {selNode.isAlive?"✓ Vivant":"✗ Mort"}
                  </span>
                </div>
                {[
                  ["Node ID",     `#${selNode.id}`,                                             C.txt],
                  ["Position",    `(${selNode.x}m, ${selNode.y}m)`,                             C.txt],
                  ["Cluster",     `C${selNode.clusterId}${selNode.isCH?" · CH":""}`,            clCol],
                  ["Énergie",     `${(selNode.energy??0).toFixed(4)} J`,                        "#ff8c00"],
                  ["Risque PEPM", `${((selNode.pepmRisk??0)*100).toFixed(1)}%`,                 rc],
                  ["Dist. Sink",  `${(selNode.distToSink??0).toFixed(1)} m`,                    C.txt],
                  ["Voisins",     `${selNode.nbCount??0} (${150}m)`,                            C.txt],
                  ["Fitness IFO", `${(selNode.fitness??0).toFixed(4)}`,                         C.purple],
                  ["Epsilon ε",   epsilon,                                                       "#ff8c00"],
                ].map(([k,v,c])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between",
                    padding:"3px 0",borderBottom:`1px solid rgba(255,255,255,.04)`,fontSize:10}}>
                    <span style={{color:C.txtDim,fontFamily:"'Space Mono',monospace"}}>{k}</span>
                    <span style={{color:c,fontFamily:"'Space Mono',monospace"}}>{v}</span>
                  </div>
                ))}
                <div style={{marginTop:7}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.txtDim,marginBottom:2}}>
                    <span>Énergie</span><span style={{color:"#ff8c00"}}>{((selNode.energyNorm??0)*100).toFixed(1)}%</span>
                  </div>
                  <div style={{background:C.muted,borderRadius:3,height:4,overflow:"hidden"}}>
                    <div style={{width:`${(selNode.energyNorm??0)*100}%`,height:"100%",borderRadius:3,
                      background:`linear-gradient(90deg,${C.red},#ff8c00,#00ff88)`,transition:"width .3s"}}/>
                  </div>
                </div>
                <div style={{marginTop:5}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.txtDim,marginBottom:2}}>
                    <span>Risque PEPM</span><span style={{color:rc}}>{((selNode.pepmRisk??0)*100).toFixed(1)}%</span>
                  </div>
                  <div style={{background:C.muted,borderRadius:3,height:4,overflow:"hidden"}}>
                    <div style={{width:`${(selNode.pepmRisk??0)*100}%`,height:"100%",borderRadius:3,
                      background:`linear-gradient(90deg,${C.green},#ff8c00,${C.red})`,transition:"width .3s"}}/>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>

        {/* Chemin → Sink */}
        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <Title accent={C.green}>Chemin → Sink</Title>
          {!selPath.ids?.length?(
            <div style={{color:C.txtDim,fontSize:10,lineHeight:1.8}}>
              Sélectionnez un nœud pour tracer son chemin greedy vers le Sink (500,500).
            </div>
          ):(
            <div>
              {/* chemin avec coordonnées, identique au HTML */}
              <div style={{fontFamily:"'Space Mono',monospace",fontSize:9,lineHeight:2,
                wordBreak:"break-all",maxHeight:120,overflowY:"auto"}}>
                {selPath.ids.map((id,i)=>{
                  const n=topoData.find(n=>n.id===id);
                  const isSrc=i===0,isCH=n?.isCH;
                  const col=isSrc||isCH?C.green:C.cyan;
                  return(
                    <span key={i}>
                      <span style={{color:col}}>#{id}({n?.x},{n?.y})</span>
                      {i<selPath.ids.length-1&&<span style={{color:C.txtDim}}> →{"\n"}</span>}
                    </span>
                  );
                })}
                <span style={{color:C.txtDim}}> → </span>
                <span style={{color:"#ffd700"}}>SINK(500,500)</span>
              </div>
              <div style={{marginTop:5,display:"flex",gap:10,fontSize:9,alignItems:"center"}}>
                <span style={{color:C.txtDim}}>{selPath.ids.length} hops</span>
                <span style={{color:selPath.delivered?C.green:"#ff8c00",fontWeight:700}}>
                  {selPath.delivered?"✓ Livré":"~ Partiel"}
                </span>
                <span style={{color:C.green,marginLeft:"auto"}}>● animé</span>
              </div>
            </div>
          )}
        </div>

        {/* Fitness IFO */}
        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <Title accent={C.purple}>Fitness IFO v2</Title>
          <div style={{fontSize:9,color:C.txtDim,marginBottom:6,lineHeight:1.7,fontFamily:"'Space Mono',monospace"}}>
            F = W1·E + W2·dSink + W3·deg − W4·rot
          </div>
          {[["W1 énergie","0.45",C.green],["W2 proximité","0.25",C.cyan],
            ["W3 densité","0.20",C.purple],["W4 rotation","0.10","#ff8c00"]].map(([l,v,c])=>(
            <div key={l} style={{marginBottom:5}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:9,marginBottom:2}}>
                <span style={{color:C.txtDim}}>{l}</span><span style={{color:c}}>{v}</span>
              </div>
              <div style={{background:C.muted,borderRadius:3,height:3}}>
                <div style={{width:`${parseFloat(v)*100}%`,height:"100%",background:c,borderRadius:3}}/>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
// ─── Drop zone générique ──────────────────────────────────────────────────────
function DropZone({label, accept, onLoad, hint}){
  const ref = useRef(null);
  const [dragging, setDragging] = useState(false);
  const read = f => {
    if(!f) return;
    const r = new FileReader();
    r.onload = e => onLoad(e.target.result, f.name);
    r.readAsText(f);
  };
  return(
    <div onDragOver={e=>{e.preventDefault();setDragging(true);}}
         onDragLeave={()=>setDragging(false)}
         onDrop={e=>{e.preventDefault();setDragging(false);read(e.dataTransfer.files[0]);}}
         onClick={()=>ref.current?.click()}
         style={{border:`2px dashed ${dragging?C.cyan:C.border}`,borderRadius:7,
           padding:"13px 16px",cursor:"pointer",
           background:dragging?`${C.cyan}08`:C.panel,transition:"all .2s",textAlign:"center",
           boxShadow:dragging?`0 0 14px ${C.cyan}30`:"none"}}>
      <div style={{fontSize:15,marginBottom:4}}>📂</div>
      <div style={{color:C.txt,fontSize:11,marginBottom:2}}>{label}</div>
      <div style={{color:C.txtDim,fontSize:9}}>{hint}</div>
      <input ref={ref} type="file" accept={accept} style={{display:"none"}}
        onChange={e=>read(e.target.files[0])}/>
    </div>
  );
}

function FileBadge({name, color=C.green, onClear}){
  return(
    <div style={{display:"inline-flex",alignItems:"center",gap:7,
      background:`${color}10`,border:`1px solid ${color}`,
      borderRadius:5,padding:"3px 10px",fontSize:9}}>
      <span style={{color}}>✓ {name}</span>
      <span onClick={e=>{e.stopPropagation();onClear();}}
        style={{color:C.txtDim,cursor:"pointer",fontSize:12,lineHeight:1}}>×</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Dashboard principal
// ─────────────────────────────────────────────────────────────────────────────
export default function FDQNDashboard(){

  // ── Données ───────────────────────────────────────────────────────────────
  const [energyData,  setEnergyData]  = useState(demoEnergy);
  const [rlData,      setRlData]      = useState(demoRL);
  const [topoData,    setTopoData]    = useState(demoTopo);
  const [summary,     setSummary]     = useState(null);    // post-sim PDR réel

  // ── Fichiers chargés ───────────────────────────────────────────────────────
  const [energyFile,  setEnergyFile]  = useState(null);
  const [rlFile,      setRlFile]      = useState(null);
  const [topoFile,    setTopoFile]    = useState(null);
  const [summaryFile, setSummaryFile] = useState(null);
  const [errors,      setErrors]      = useState({});

  // ── UI ─────────────────────────────────────────────────────────────────────
  const [tab,         setTab]         = useState("energy");
  const [showPanel,   setShowPanel]   = useState(false);

  // ── Live mode ──────────────────────────────────────────────────────────────
  const [live,        setLive]        = useState(false);
  const [liveInt,     setLiveInt]     = useState(5);
  const [cd,          setCd]          = useState(5);
  const energyRef  = useRef(null);
  const rlRef      = useRef(null);
  const topoRef    = useRef(null);
  const summaryRef = useRef(null);
  const timerRef   = useRef(null);
  const cdRef      = useRef(null);

  // ── Chargeurs ─────────────────────────────────────────────────────────────
  const loadEnergy = useCallback((text, name) => {
    try {
      const p = parseEnergyCSV(text);
      if (!p.length) throw new Error("Aucune ligne valide.");
      setEnergyData(p); setEnergyFile(name);
      setErrors(e=>({...e,energy:null}));
    } catch(err) { setErrors(e=>({...e,energy:err.message})); }
  },[]);

  const loadRL = useCallback((text, name) => {
    try {
      const p = parseRLJson(text);
      if (!p.series.length) throw new Error("Historique vide.");
      setRlData(p); setRlFile(name);
      setErrors(e=>({...e,rl:null}));
    } catch(err) { setErrors(e=>({...e,rl:err.message})); }
  },[]);

  const loadTopo = useCallback((text, name) => {
    try {
      const p = parseTopologyCSV(text);
      if (!p.length) throw new Error("Aucun nœud trouvé.");
      setTopoData(p); setTopoFile(name);
      setErrors(e=>({...e,topo:null}));
    } catch(err) { setErrors(e=>({...e,topo:err.message})); }
  },[]);

  const loadSummary = useCallback((text, name) => {
    try {
      const p = parseSummaryCSV(text);
      if (p.PDR_pct === undefined) throw new Error("PDR_pct introuvable.");
      setSummary(p); setSummaryFile(name);
      setErrors(e=>({...e,summary:null}));
    } catch(err) { setErrors(e=>({...e,summary:err.message})); }
  },[]);

  // ── Refresh live ──────────────────────────────────────────────────────────
  const refreshAll = useCallback(() => {
    const reread = (ref, loader) => {
      if (!ref.current) return;
      const r = new FileReader();
      r.onload = e => loader(e.target.result, ref.current.name);
      r.readAsText(ref.current);
    };
    reread(energyRef,  loadEnergy);
    reread(rlRef,      loadRL);
    reread(topoRef,    loadTopo);
    reread(summaryRef, loadSummary);
  },[loadEnergy,loadRL,loadTopo,loadSummary]);

  const toggleLive = useCallback(() => {
    if (live) {
      clearInterval(timerRef.current); clearInterval(cdRef.current);
      setLive(false); return;
    }
    setLive(true); setCd(liveInt);
    timerRef.current = setInterval(()=>{ refreshAll(); setCd(liveInt); }, liveInt*1000);
    cdRef.current    = setInterval(()=>setCd(c=>Math.max(0,c-1)), 1000);
  },[live,liveInt,refreshAll]);

  useEffect(()=>()=>{clearInterval(timerRef.current);clearInterval(cdRef.current);},[]);

  // ── Données dérivées ───────────────────────────────────────────────────────
  const last   = energyData[energyData.length-1] || {};
  const fndRow = energyData.find(r=>r.fnd>0);
  const hndRow = energyData.find(r=>r.hnd>0);

  // PDR : priorité summary (post-sim, FlowMonitor) sinon energy.csv (toujours 0 en vrai)
  const pdrReal    = summary?.PDR_pct ?? null;
  const pdrDisplay = pdrReal !== null ? pdrReal : (last.pdr||0);
  const pdrLabel   = pdrReal !== null
    ? `${pdrReal.toFixed(1)}% (FlowMonitor)`
    : (energyFile ? "0% (post-sim uniquement)" : `${(last.pdr||0).toFixed(1)}% (démo)`);

  const anyReal = energyFile || rlFile || topoFile || summaryFile;

  const TABS = [
    {id:"energy",   label:"Énergie & Vie",   fileOk:!!energyFile},
    {id:"rl",       label:"Apprentissage RL", fileOk:!!rlFile},
    {id:"topology", label:"Topologie",        fileOk:!!topoFile},
    {id:"pepm",     label:"PEPM & Fédération",fileOk:!!energyFile},
  ];

  return(
    <div style={{background:C.bg,minHeight:"100vh",fontFamily:"'Space Mono',monospace",color:C.txt}}>

      {/* ── HEADER ─────────────────────────────────────────────────────────── */}
      <div style={{background:`linear-gradient(135deg,${C.bg2} 0%,#071428 100%)`,
        borderBottom:`1px solid ${C.border}`,padding:"13px 20px"}}>

        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",gap:10,flexWrap:"wrap"}}>

          {/* Titre */}
          <div>
            <div style={{display:"flex",alignItems:"center",gap:10}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:C.green,
                boxShadow:`0 0 10px ${C.green}`,animation:"pulse 2s infinite"}}/>
              <span style={{color:C.cyan,fontSize:16,fontWeight:700,letterSpacing:3}}>FDQN-TE+</span>
              <span style={{color:C.txtDim,fontSize:10,letterSpacing:2}}>DASHBOARD v2.1</span>
              {!anyReal && <span style={{color:C.amber,fontSize:9,border:`1px solid ${C.amber}`,borderRadius:4,padding:"1px 6px"}}>DÉMO</span>}
            </div>
            <div style={{color:C.txtDim,fontSize:9,marginTop:3,letterSpacing:1}}>
              NS-3.39 · IFO + ADDQN + PEPM + FedMeta-DRL
            </div>
          </div>

          {/* Contrôles live */}
          <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
            <div style={{display:"flex",alignItems:"center",gap:7,
              background:live?`${C.green}12`:`${C.muted}20`,
              border:`1px solid ${live?C.green:C.border}`,borderRadius:20,padding:"4px 11px",fontSize:9}}>
              <div style={{width:6,height:6,borderRadius:"50%",background:live?C.green:C.txtDim,
                boxShadow:live?`0 0 8px ${C.green}`:"none",animation:live?"pulse 1.5s infinite":"none"}}/>
              <span style={{color:live?C.green:C.txtDim}}>{live?`LIVE · ${cd}s`:"hors-ligne"}</span>
            </div>

            {!live && (
              <select value={liveInt} onChange={e=>setLiveInt(+e.target.value)} style={{
                background:C.panel,border:`1px solid ${C.border}`,color:C.txt,
                borderRadius:6,padding:"3px 7px",fontSize:9,cursor:"pointer",fontFamily:"'Space Mono',monospace"}}>
                {[3,5,10,30,60].map(s=><option key={s} value={s}>{s}s</option>)}
              </select>
            )}

            <button onClick={toggleLive} style={{
              background:live?`${C.red}15`:`${C.green}15`,
              border:`1px solid ${live?C.red:C.green}`,color:live?C.red:C.green,
              borderRadius:6,padding:"4px 12px",fontSize:9,cursor:"pointer",
              fontFamily:"'Space Mono',monospace",letterSpacing:1}}>
              {live?"⏹ STOP":"▶ LIVE"}
            </button>

            <button onClick={()=>setShowPanel(v=>!v)} style={{
              background:showPanel?`${C.cyan}18`:`${C.cyan}10`,
              border:`1px solid ${C.cyan}`,color:C.cyan,
              borderRadius:6,padding:"4px 12px",fontSize:9,cursor:"pointer",
              fontFamily:"'Space Mono',monospace",letterSpacing:1}}>
              {showPanel?"▲ Fichiers":"📂 Fichiers"}
            </button>
          </div>

          {/* KPIs header */}
          <div style={{display:"flex",gap:16}}>
            {[
              ["FND", fndRow?`${fndRow.fnd}s`:"—", C.amber],
              ["HND", hndRow?`${hndRow.hnd}s`:"—", C.purple],
              ["PDR", pdrReal!==null?`${pdrReal.toFixed(1)}%`:energyFile?"post-sim":`${(last.pdr||0).toFixed(1)}%`,
                      pdrReal!==null?(pdrReal>80?C.green:C.red):energyFile?C.amber:C.cyan],
            ].map(([l,v,c])=>(
              <div key={l} style={{textAlign:"center"}}>
                <div style={{color:C.txtDim,fontSize:9,letterSpacing:2}}>{l}</div>
                <div style={{color:c,fontSize:16,fontWeight:700}}>{v}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── PANNEAU FICHIERS ───────────────────────────────────────────── */}
        {showPanel && (
          <div style={{marginTop:13}}>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:10,marginBottom:10}}>

              {/* Energy CSV */}
              <div>
                <div style={{color:C.txtDim,fontSize:9,marginBottom:4,letterSpacing:1}}>
                  ÉNERGIE · <code style={{color:C.cyan}}>fdqnte_energy.csv</code>
                </div>
                {energyFile
                  ? <FileBadge name={energyFile} color={C.cyan} onClear={()=>{setEnergyData(demoEnergy());setEnergyFile(null);energyRef.current=null;}}/>
                  : <DropZone label="fdqnte_energy.csv" accept=".csv,.txt"
                      hint="Énergie, FND, HND, PEPM (PDR=0 ici)"
                      onLoad={(t,n)=>{loadEnergy(t,n);const b=new Blob([t],{type:"text/plain"});b.name=n;energyRef.current=b;}}/>
                }
                {errors.energy && <div style={{color:C.red,fontSize:9,marginTop:3}}>⚠ {errors.energy}</div>}
              </div>

              {/* Summary CSV — PDR réel */}
              <div>
                <div style={{color:C.txtDim,fontSize:9,marginBottom:4,letterSpacing:1}}>
                  PDR RÉEL · <code style={{color:C.green}}>fdqnte_summary.csv</code>
                </div>
                {summaryFile
                  ? <FileBadge name={summaryFile} color={C.green} onClear={()=>{setSummary(null);setSummaryFile(null);summaryRef.current=null;}}/>
                  : <DropZone label="fdqnte_summary.csv" accept=".csv,.txt"
                      hint="PDR FlowMonitor, TxPackets, RxPackets"
                      onLoad={(t,n)=>{loadSummary(t,n);const b=new Blob([t],{type:"text/plain"});b.name=n;summaryRef.current=b;}}/>
                }
                {errors.summary && <div style={{color:C.red,fontSize:9,marginTop:3}}>⚠ {errors.summary}</div>}
              </div>

              {/* RL JSON */}
              <div>
                <div style={{color:C.txtDim,fontSize:9,marginBottom:4,letterSpacing:1}}>
                  RL HISTORY · <code style={{color:C.purple}}>fdqnte_rl_history.json</code>
                </div>
                {rlFile
                  ? <FileBadge name={rlFile} color={C.purple} onClear={()=>{setRlData(demoRL());setRlFile(null);rlRef.current=null;}}/>
                  : <DropZone label="fdqnte_rl_history.json" accept=".json"
                      hint="Reward, Loss, ε — écrit ttes les 30s"
                      onLoad={(t,n)=>{loadRL(t,n);const b=new Blob([t],{type:"text/plain"});b.name=n;rlRef.current=b;}}/>
                }
                {errors.rl && <div style={{color:C.red,fontSize:9,marginTop:3}}>⚠ {errors.rl}</div>}
              </div>

              {/* Topology CSV */}
              <div>
                <div style={{color:C.txtDim,fontSize:9,marginBottom:4,letterSpacing:1}}>
                  TOPOLOGIE · <code style={{color:C.amber}}>fdqnte_topology_final.csv</code>
                </div>
                {topoFile
                  ? <FileBadge name={topoFile} color={C.amber} onClear={()=>{setTopoData(demoTopo());setTopoFile(null);topoRef.current=null;}}/>
                  : <DropZone label="fdqnte_topology_final.csv" accept=".csv,.txt"
                      hint="NodeId,X,Y,EnergyNorm,IsClusterHead..."
                      onLoad={(t,n)=>{loadTopo(t,n);const b=new Blob([t],{type:"text/plain"});b.name=n;topoRef.current=b;}}/>
                }
                {errors.topo && <div style={{color:C.red,fontSize:9,marginTop:3}}>⚠ {errors.topo}</div>}
              </div>
            </div>

            {/* Explication PDR */}
            <div style={{background:`${C.amber}08`,border:`1px solid ${C.amber}40`,
              borderRadius:6,padding:"7px 12px",fontSize:9,color:C.txtDim,lineHeight:1.8}}>
              <span style={{color:C.amber}}>⚠ Pourquoi PDR = 0 dans energy.csv ?</span><br/>
              NS-3 calcule le PDR via FlowMonitor <strong style={{color:C.txt}}>après</strong> la simulation.
              Les lignes de <code style={{color:C.cyan}}>fdqnte_energy.csv</code> ont donc <code>PDR_pct=0.0</code> pendant la sim.<br/>
              Le PDR réel est dans <code style={{color:C.green}}>fdqnte_summary.csv</code> → charge ce fichier pour l'afficher.
            </div>
          </div>
        )}
      </div>

      {/* ── KPI CARDS ──────────────────────────────────────────────────────── */}
      <div style={{padding:"14px 20px 0",display:"grid",gridTemplateColumns:"repeat(6,1fr)",gap:9}}>
        <Card label="Vivants"    value={last.alive||0}   unit="/300" color={C.green}  sub={`${last.dead||0} morts`}/>
        <Card label="E moy"      value={(last.energy||0).toFixed(3)} unit="J" color={C.cyan} sub={`min ${(last.energyMin||0).toFixed(3)}J`}/>
        <Card label="Steps RL"   value={(rlData.stats?.global_step||last.rlSteps||0).toLocaleString()} color={C.purple} sub={`fed: ${last.fedRound||0}`}/>
        <Card label="IFO rounds" value={last.ifoRound||0}            color={C.amber}  sub="re-clusters"/>
        <Card label="PEPM@risk"  value={last.atRisk||0}  unit=" nœuds" color={(last.atRisk||0)>50?C.red:C.green} sub="seuil 0.70"/>
        <Card label="PDR"
          value={pdrReal!==null ? `${pdrReal.toFixed(1)}` : energyFile ? "—" : `${(last.pdr||0).toFixed(1)}`}
          unit="%"
          color={pdrReal!==null?(pdrReal>80?C.green:C.red):energyFile?C.muted:C.amber}
          sub={pdrReal!==null?"FlowMonitor ✓":energyFile?"charger summary.csv":"démo"}/>
      </div>

      {/* ── TABS ───────────────────────────────────────────────────────────── */}
      <div style={{padding:"14px 20px 0"}}>
        <div style={{display:"flex",gap:0,borderBottom:`1px solid ${C.border}`}}>
          {TABS.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              background:tab===t.id?C.panel:"transparent",border:"none",
              borderBottom:tab===t.id?`2px solid ${C.cyan}`:"2px solid transparent",
              color:tab===t.id?C.txt:C.txtDim,padding:"8px 14px",cursor:"pointer",
              fontSize:10,letterSpacing:1,fontFamily:"'Space Mono',monospace",transition:"all .2s"}}>
              {t.label}
              {t.fileOk && <span style={{display:"inline-block",width:5,height:5,borderRadius:"50%",
                background:C.green,boxShadow:`0 0 5px ${C.green}`,marginLeft:6,verticalAlign:"middle"}}/>}
            </button>
          ))}
        </div>
      </div>

      {/* ── CONTENU ONGLETS ────────────────────────────────────────────────── */}
      <div style={{padding:"14px 20px 24px"}}>

        {/* ── ENERGY ── */}
        {tab==="energy" && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.cyan}>Énergie résiduelle moyenne</Title>
              <ResponsiveContainer width="100%" height={215}>
                <AreaChart data={energyData}>
                  <defs>
                    <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.cyan}   stopOpacity={0.3}/><stop offset="95%" stopColor={C.cyan}   stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="emg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.purple} stopOpacity={0.15}/><stop offset="95%" stopColor={C.purple} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="time" stroke={C.txtDim} tick={{fontSize:9}} label={{value:"t(s)",position:"insideRight",fill:C.txtDim,fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,2.2]}/>
                  <Tooltip {...tt} formatter={(v,n)=>[`${v} J`,n]}/>
                  <Legend wrapperStyle={{fontSize:9,color:C.txtDim}}/>
                  <Area type="monotone" dataKey="energy"    name="E moy" stroke={C.cyan}   strokeWidth={2} fill="url(#eg)"/>
                  <Area type="monotone" dataKey="energyMin" name="E min" stroke={C.purple} strokeWidth={1} strokeDasharray="4 2" fill="url(#emg)"/>
                  {fndRow&&<ReferenceLine x={fndRow.fnd} stroke={C.amber} strokeDasharray="5 3" label={{value:"FND",fill:C.amber,fontSize:9}}/>}
                  {hndRow&&<ReferenceLine x={hndRow.hnd} stroke={C.red}   strokeDasharray="5 3" label={{value:"HND",fill:C.red,  fontSize:9}}/>}
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.green}>Nœuds vivants / morts</Title>
              <ResponsiveContainer width="100%" height={215}>
                <AreaChart data={energyData}>
                  <defs>
                    <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.green} stopOpacity={0.25}/><stop offset="95%" stopColor={C.green} stopOpacity={0}/>
                    </linearGradient>
                    <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.red}   stopOpacity={0.25}/><stop offset="95%" stopColor={C.red}   stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="time" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,320]}/>
                  <Tooltip {...tt}/>
                  <Legend wrapperStyle={{fontSize:9,color:C.txtDim}}/>
                  <Area type="monotone" dataKey="alive" name="Vivants" stroke={C.green} strokeWidth={2} fill="url(#ag)"/>
                  <Area type="monotone" dataKey="dead"  name="Morts"   stroke={C.red}   strokeWidth={2} fill="url(#dg)"/>
                  <ReferenceLine y={150} stroke={C.red} strokeDasharray="4 2" label={{value:"seuil HND",fill:C.red,fontSize:9}}/>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* PDR avec explication */}
            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.amber}>PDR</Title>
              {energyFile && !summaryFile && (
                <div style={{marginBottom:10,background:`${C.amber}10`,border:`1px solid ${C.amber}40`,
                  borderRadius:6,padding:"6px 10px",fontSize:9,color:C.txtDim,lineHeight:1.7}}>
                  <span style={{color:C.amber}}>ℹ PDR=0 dans energy.csv</span> — c'est normal.<br/>
                  Charge <code style={{color:C.green}}>fdqnte_summary.csv</code> pour le PDR FlowMonitor réel.
                </div>
              )}
              {pdrReal !== null ? (
                <div style={{textAlign:"center",padding:"20px 0"}}>
                  <div style={{color:C.txtDim,fontSize:10,marginBottom:8}}>PDR FlowMonitor (post-simulation)</div>
                  <div style={{color:pdrReal>80?C.green:C.red,fontSize:52,fontWeight:700,fontFamily:"'Space Mono',monospace",lineHeight:1}}>
                    {pdrReal.toFixed(1)}%
                  </div>
                  <div style={{color:C.txtDim,fontSize:10,marginTop:8}}>
                    Tx: {summary?.TxPackets?.toLocaleString()||"—"} · Rx: {summary?.RxPackets?.toLocaleString()||"—"}
                  </div>
                  <div style={{background:C.muted,borderRadius:4,height:8,overflow:"hidden",marginTop:12}}>
                    <div style={{width:`${Math.min(100,pdrReal)}%`,height:"100%",borderRadius:4,
                      background:`linear-gradient(90deg,${C.green},${pdrReal>80?C.green:C.amber})`,
                      boxShadow:`0 0 8px ${pdrReal>80?C.green:C.amber}`,transition:"width .5s ease"}}/>
                  </div>
                </div>
              ) : (
                <ResponsiveContainer width="100%" height={185}>
                  <LineChart data={energyData}>
                    <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                    <XAxis dataKey="time" stroke={C.txtDim} tick={{fontSize:9}}/>
                    <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,105]}/>
                    <Tooltip {...tt} formatter={v=>[`${v}%`,"PDR"]}/>
                    <Line type="monotone" dataKey="pdr" stroke={C.amber} strokeWidth={2} dot={false}/>
                    <ReferenceLine y={80} stroke={C.green} strokeDasharray="4 2" label={{value:"cible 80%",fill:C.green,fontSize:9}}/>
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.red}>PEPM — Nœuds à risque</Title>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:5}}>
                <span style={{color:C.txtDim,fontSize:10}}>Fraction critique</span>
                <span style={{color:C.red,fontSize:11,fontFamily:"'Space Mono',monospace"}}>{last.atRisk||0} / 300</span>
              </div>
              <div style={{background:C.muted,borderRadius:4,height:6,overflow:"hidden",marginBottom:10}}>
                <div style={{width:`${Math.min(100,((last.atRisk||0)/300)*100)}%`,height:"100%",borderRadius:4,
                  background:`linear-gradient(90deg,${C.green},${C.red})`,transition:"width .5s ease"}}/>
              </div>
              <ResponsiveContainer width="100%" height={145}>
                <AreaChart data={energyData}>
                  <defs>
                    <linearGradient id="rg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.red} stopOpacity={0.3}/><stop offset="95%" stopColor={C.red} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="time" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}}/>
                  <Tooltip {...tt}/>
                  <Area type="monotone" dataKey="atRisk" stroke={C.red} strokeWidth={2} fill="url(#rg)" name="@risque"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* ── RL ── */}
        {tab==="rl" && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
            {!rlFile && (
              <div style={{gridColumn:"1/-1",background:`${C.purple}10`,border:`1px solid ${C.purple}40`,
                borderRadius:8,padding:"9px 14px",fontSize:9,color:C.txtDim,display:"flex",alignItems:"center",gap:10}}>
                <span style={{color:C.purple}}>ℹ</span>
                Données démo — charge <code style={{color:C.purple}}>fdqnte_rl_history.json</code> via
                <strong style={{color:C.txt}}> 📂 Fichiers</strong>.
                Le fichier est écrit par <code>rl_server.py</code> toutes les <strong style={{color:C.txt}}>30s</strong>.
              </div>
            )}

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.purple}>Récompense ADDQN</Title>
              <ResponsiveContainer width="100%" height={230}>
                <AreaChart data={rlData.series}>
                  <defs>
                    <linearGradient id="rwg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.purple} stopOpacity={0.35}/><stop offset="95%" stopColor={C.purple} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="step" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,1]}/>
                  <Tooltip {...tt}/>
                  <Area type="monotone" dataKey="reward" stroke={C.purple} strokeWidth={2} fill="url(#rwg)" name="Reward"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.cyan}>Décroissance ε-greedy</Title>
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={rlData.series}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="step" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,1.1]}/>
                  <Tooltip {...tt}/>
                  <Line type="monotone" dataKey="epsilon" stroke={C.cyan} strokeWidth={2} dot={false} name="ε"/>
                  <ReferenceLine y={0.05} stroke={C.green} strokeDasharray="4 2" label={{value:"ε_min",fill:C.green,fontSize:9}}/>
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.amber}>TD-Loss</Title>
              <ResponsiveContainer width="100%" height={195}>
                <AreaChart data={rlData.series}>
                  <defs>
                    <linearGradient id="lg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.amber} stopOpacity={0.25}/><stop offset="95%" stopColor={C.amber} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="step" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}}/>
                  <Tooltip {...tt} formatter={v=>[v.toFixed(5),"Loss"]}/>
                  <Area type="monotone" dataKey="loss" stroke={C.amber} strokeWidth={1.5} fill="url(#lg)" name="TD-Loss"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.green}>Config & Stats {rlFile?"(réelles)":"(démo)"}</Title>
              {[
                ["γ",              rlData.config?.gamma              ?? 0.99],
                ["LR",             rlData.config?.lr                 ?? "1e-3"],
                ["ε decay",        rlData.config?.epsilon_decay      ?? 0.995],
                ["Fed period",     rlData.config?.fed_period         ?? 50],
                ["Steps total",    (rlData.stats?.global_step        ?? 0).toLocaleString()],
                ["Actions req.",   (rlData.stats?.actions_requested  ?? 0).toLocaleString()],
                ["Rewards reçus",  (rlData.stats?.rewards_received   ?? 0).toLocaleString()],
                ["PEPM queries",   (rlData.stats?.pepm_queries       ?? 0).toLocaleString()],
              ].map(([k,v])=>(
                <div key={k} style={{display:"flex",justifyContent:"space-between",
                  borderBottom:`1px solid ${C.border}`,padding:"4px 0",fontSize:10}}>
                  <span style={{color:C.txtDim}}>{k}</span>
                  <span style={{color:C.green,fontFamily:"'Space Mono',monospace"}}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── TOPOLOGY ── */}
        {tab==="topology" && <TopoTab topoData={topoData} topoFile={topoFile} energyData={energyData} rlData={rlData} summary={summary}/>}

        {/* ── PEPM & FED ── */}
        {tab==="pepm" && (
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.red}>PEPM risk(E) — formule v2</Title>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={Array.from({length:100},(_,i)=>{
                  const e=i/100*2, te=0.6;
                  const risk=e>=te?0.15*(1-(e-te)/(2-te)):Math.min(1,0.15+0.85*(1-e/te));
                  return{e:+e.toFixed(2),risk:+risk.toFixed(3),seuil:0.7};
                })}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="e" stroke={C.txtDim} tick={{fontSize:9}} label={{value:"E(J)",position:"insideRight",fill:C.txtDim,fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}} domain={[0,1.1]}/>
                  <Tooltip {...tt}/>
                  <Line type="monotone" dataKey="risk"  stroke={C.red}   strokeWidth={2} dot={false} name="risk"/>
                  <Line type="monotone" dataKey="seuil" stroke={C.amber} strokeWidth={1} strokeDasharray="5 3" dot={false} name="seuil 0.70"/>
                </LineChart>
              </ResponsiveContainer>
              <div style={{fontSize:9,color:C.txtDim,marginTop:8,lineHeight:1.8}}>
                <span style={{color:C.cyan}}>Zone sûre</span> (E&gt;0.6J) : risk = 0.15×(1−(E−0.6)/1.4)<br/>
                <span style={{color:C.red}}>Zone critique</span> (E≤0.6J) : risk = 0.15+0.85×(1−E/0.6)<br/>
                <span style={{color:C.amber}}>→ seuil 0.70 dès E≈0.18J</span>
              </div>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.purple}>Rounds fédérés</Title>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={energyData}>
                  <defs>
                    <linearGradient id="fg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.purple} stopOpacity={0.3}/><stop offset="95%" stopColor={C.purple} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
                  <XAxis dataKey="time" stroke={C.txtDim} tick={{fontSize:9}}/>
                  <YAxis stroke={C.txtDim} tick={{fontSize:9}}/>
                  <Tooltip {...tt}/>
                  <Area type="monotone" dataKey="fedRound" stroke={C.purple} strokeWidth={2} fill="url(#fg)" name="Fed rounds"/>
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.amber}>Re-clustering adaptatif</Title>
              {[
                [C.green,  "Normal",       "toutes les 100s (round pair)"],
                [C.red,    "URGENT-PEPM",  "si CH a pepmRisk ≥ 0.80"],
                [C.purple, "Rotation CH",  "pénalité W4=0.10 dans fitness IFO"],
              ].map(([c,t,d])=>(
                <div key={t} style={{borderLeft:`3px solid ${c}`,paddingLeft:10,marginBottom:11}}>
                  <div style={{color:c,fontSize:10,fontWeight:700}}>{t}</div>
                  <div style={{color:C.txtDim,fontSize:9}}>{d}</div>
                </div>
              ))}
            </div>

            <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:14}}>
              <Title accent={C.cyan}>Récapitulatif fichiers</Title>
              {[
                [energyFile,  "fdqnte_energy.csv",         C.cyan,   "Énergie, FND/HND, PEPM, Fédération"],
                [summaryFile, "fdqnte_summary.csv",        C.green,  "PDR FlowMonitor (post-simulation)"],
                [rlFile,      "fdqnte_rl_history.json",    C.purple, "Reward, Loss, ε — écrit ttes les 30s"],
                [topoFile,    "fdqnte_topology_final.csv", C.amber,  "Positions, CH, EnergyNorm, fitness"],
              ].map(([loaded,name,c,desc])=>(
                <div key={name} style={{display:"flex",gap:9,padding:"6px 0",borderBottom:`1px solid ${C.border}`}}>
                  <span style={{color:loaded?c:C.muted,fontSize:13}}>{loaded?"●":"○"}</span>
                  <div>
                    <div style={{fontSize:9,fontFamily:"'Space Mono',monospace",color:loaded?c:C.txtDim}}>{name}</div>
                    <div style={{fontSize:9,color:C.txtDim}}>{desc}</div>
                    {loaded && <div style={{fontSize:9,color:C.green}}>✓ {loaded}</div>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
        @keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.85)}}
        *{box-sizing:border-box}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px}
        button:hover{opacity:.85} select:focus{outline:none} option{background:${C.panel}}
      `}</style>
    </div>
  );
}
