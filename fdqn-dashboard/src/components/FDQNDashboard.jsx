import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine, ComposedChart
} from "recharts";
import { Title, KPI, Panel, DropZone, Badge, PDRTooltip } from "./UI";
import { C, tt } from "../styles/colors";
import { parseEnergy, parseSummary, parseTopo, parseRL, parseRouting } from "../utils/parsers";
import { clCol } from "../utils/clusterColors";



// ── Parser fdqnte_routing.csv ──────────────────────────────────────────────────



// ── Cluster colors (pour TopoCanvas et TopoTab) ──────────────────────────────
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
          borderRadius:5,padding:"4px 7px",fontSize:9,cursor:"pointer",fontFamily:"'JetBrains Mono',monospace"}}>
          {VIEWS.map(([v,l])=><option key={v} value={v}>{l}</option>)}
        </select>
        {[["links","Liens"],["labels","Labels CH"],["paths","Routes"],["radio","Portée"]].map(([k,l])=>(
          <button key={k} onClick={()=>setFlags(f=>({...f,[k]:!f[k]}))} style={{
            background:flags[k]?`${C.cyan}15`:C.panel,
            border:`1px solid ${flags[k]?C.cyan:C.border}`,color:flags[k]?C.cyan:C.dim,
            borderRadius:5,padding:"4px 9px",fontSize:9,cursor:"pointer",fontFamily:"'JetBrains Mono',monospace",
            letterSpacing:.5}}>
            {l}
          </button>
        ))}
        <button onClick={resetView} style={{
          background:C.panel,border:`1px solid ${C.border}`,color:C.dim,
          borderRadius:5,padding:"4px 9px",fontSize:9,cursor:"pointer",fontFamily:"'JetBrains Mono',monospace"}}>
          ⌖ Reset
        </button>
      </div>
      {/* canvas */}
      <div style={{flex:1,position:"relative",minHeight:360}}>
        <canvas style={{display:"block",width:"100%",cursor:"crosshair",borderRadius:6,border:`1px solid ${C.border}`}}
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
              borderRadius:4,padding:"3px 8px",fontFamily:"'JetBrains Mono',monospace",fontSize:9,color:C.dim}}>
              {t}
            </div>
          ))}
          <div style={{background:"rgba(6,13,24,.88)",border:`1px solid ${C.border}`,
            borderRadius:4,padding:"3px 8px",fontFamily:"'JetBrains Mono',monospace",fontSize:9,color:C.dim,
            display:"flex",alignItems:"center",gap:6}}>
            <div style={{width:`${Math.max(12,hud.scPx)}px`,height:2,background:C.dim,borderRadius:1}}/>
            100m
          </div>
        </div>
        {/* tooltip */}
        {tip&&(
          <div style={{position:"fixed",left:tip.x+15,top:tip.y-8,
            background:"#0a1522",border:`1px solid ${C.cyan}`,borderRadius:7,
            padding:"9px 13px",fontFamily:"'JetBrains Mono',monospace",fontSize:10,
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
              <div key={k} style={{display:"flex",justifyContent:"space-between",gap:14,padding:"2px 0",color:C.dim}}>
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
export default function FDQNDashboard() {
  const [energy, setEnergy] = useState(null);
  const [summary, setSummary] = useState(null);
  const [rl, setRl] = useState(null);
  const [topoI, setTopoI] = useState(null);
  const [topoF, setTopoF] = useState(null);
  const [routing, setRouting] = useState(null);
  const [tab, setTab] = useState("energy");
  const [panel, setPanel] = useState(false);
  const [files, setFiles] = useState({});
  const [selNode, setSelNode] = useState(null);
  const [selPath, setSelPath] = useState({ids:[],delivered:false});
  const [selId,   setSelId]   = useState(null);
  const [topoMode, setTopoMode] = useState("final");

  /* ── Chargeurs ── */
  const load = (key, parser, setter) => (txt, name) => {
    try {
      const result = parser(txt);
      setter(result);
      setFiles(f => ({ ...f, [key]: name }));
      console.log(`Chargé ${key}:`, result ? "succès" : "vide");
    } catch (e) {
      console.error(`Erreur parsing ${name}:`, e);
      alert(`Erreur parsing ${name}: ${e.message}`);
    }
  };

  /* ── Données dérivées ── */
  const rows = energy || [];
  const rlRows = rl?.history || [];

  const last = rows[rows.length - 1] || {};
  const fndT = summary?.FND_t ?? rows.find(r => r.fnd > 0)?.fnd ?? rl?.metrics?.fnd_time_s ?? 0;
  const hndT = summary?.HND_t ?? rows.find(r => r.hnd > 0)?.hnd ?? rl?.metrics?.hnd_time_s ?? 0;
  const lndT = summary?.LND_t ?? 0;
  const pdrRL = summary?.PDR_RL_pct ?? last.pdrRL ?? rl?.metrics?.avg_pdr_RL_pct ?? 0;
  const N = summary?.N ?? 300;
  const alive = summary?.AliveNodes ?? last.alive ?? N;
  const dead = summary?.DeadNodes ?? last.dead ?? 0;

  const topo = topoMode === "final" ? (topoF || topoI || []) : (topoI || []);

  /* ── Données reward depuis RL history ── */
  const rewardData = useMemo(() => rlRows.map(r => ({ time: r.time, mean: r.rewardMean, min: r.rewardMin, max: r.rewardMax, atRisk: r.atRisk })), [rlRows]);

  const TABS = [
    { id: "energy", label: "Énergie & Durée de vie" },
    { id: "pdr", label: "PDR & Délai" },
    { id: "rl", label: "Apprentissage RL" },
    { id: "topo", label: "Topologie" },
    { id: "routing", label: "Routage RL" },

  ];

  const pdrColor = pdrRL >= 90 ? C.green : pdrRL >= 80 ? C.amber : C.red;

  // Animation CSS
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
      }
    `;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // Statistiques pour la topologie
  const clusterStats = useMemo(() => {
    if (!topo.length && !topoI) return { clusters: [], topCH: [], pepmBins: [], sourceLabel: "" };

    // Pour les graphiques analytiques: si la topo courante n'a pas de CH vivants
    // (ex: topologie finale où tous les CH sont morts), utiliser la topologie initiale
    const hasAliveCH = topo.filter(n => n.isCH && n.isAlive).length > 0;
    const topoSrc = hasAliveCH ? topo : (topoI || topo);
    const sourceLabel = hasAliveCH ? "" : " (données initiales)";

    const clusterMap = {};
    topoSrc.forEach(n => {
      if (!clusterMap[n.clusterId]) {
        clusterMap[n.clusterId] = { ch: null, members: [], energy: [] };
      }
      if (n.isCH) clusterMap[n.clusterId].ch = n;
      if (n.isAlive) {
        if (!n.isCH) clusterMap[n.clusterId].members.push(n);
        clusterMap[n.clusterId].energy.push(n.energy);
      }
    });

    const clusters = Object.entries(clusterMap)
      .filter(([, c]) => c.ch !== null && c.ch.isAlive && c.members.length > 0)
      .map(([id, c]) => ({
        id: `#${c.ch.id}`,
        chId: c.ch.id,
        members: c.members.length,
        energyMoy: c.energy.length ? +(c.energy.reduce((s, e) => s + e, 0) / c.energy.length).toFixed(3) : 0,
        energyMin: c.energy.length ? +Math.min(...c.energy).toFixed(3) : 0,
        fitness:   +(c.ch.fitness ?? 0).toFixed(3),
      }))
      .sort((a, b) => b.members - a.members)
      .slice(0, 24);

    const topCH = topoSrc
      .filter(n => n.isCH && n.isAlive)
      .sort((a, b) => (b.fitness ?? 0) - (a.fitness ?? 0))
      .slice(0, 8);

    const pepmBins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((lo, i, arr) => ({
      range: `${(lo * 100).toFixed(0)}-${((arr[i + 1] || 1) * 100).toFixed(0)}%`,
      count: topoSrc.filter(n => n.isAlive && n.pepmRisk >= lo && n.pepmRisk < (arr[i + 1] || 1.01)).length,
      danger: lo >= 0.7,
    }));

    return { clusters, topCH, pepmBins, sourceLabel };
  }, [topo, topoI]);

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'JetBrains Mono', monospace", color: C.txt }}>

      {/* ── HEADER ── */}
      <div style={{ background: `linear-gradient(135deg,#060d1a 0%,#050810 100%)`, borderBottom: `1px solid ${C.border}`, padding: "11px 20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>

          {/* Logo */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: C.green, boxShadow: `0 0 10px ${C.green}`, animation: "pulse 2s infinite" }} />
              <span style={{ color: C.cyan, fontSize: 15, fontWeight: 700, letterSpacing: 3 }}>FDQN-TE+</span>
              <span style={{ color: C.dim, fontSize: 9, letterSpacing: 2 }}>WSN DASHBOARD</span>
              {!files.energy && <span style={{ color: C.amber, fontSize: 8, border: `1px solid ${C.amber}40`, borderRadius: 3, padding: "1px 5px" }}>SANS DONNÉES</span>}
            </div>
            <div style={{ color: C.dim, fontSize: 8, marginTop: 2, letterSpacing: 1 }}>IFO + ADDQN + PEPM + FedMeta-DRL</div>
          </div>

          {/* KPIs header */}
          <div style={{ display: "flex", gap: 20 }}>
            {[
              ["FND", fndT ? `${fndT}s` : "—", C.amber],
              ["HND", hndT ? `${hndT}s` : "—", C.purple],
              ["PDR RL", pdrRL ? `${pdrRL.toFixed(1)}%` : "—", pdrColor],
              ["Vivants", alive ? `${alive}/${N}` : "—", C.green],
            ].map(([l, v, c]) => (
              <div key={l} style={{ textAlign: "center" }}>
                <div style={{ color: C.dim, fontSize: 8, letterSpacing: 2 }}>{l}</div>
                <div style={{ color: c, fontSize: 15, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace" }}>{v}</div>
              </div>
            ))}
          </div>

          {/* Boutons */}
          <div style={{ display: "flex", gap: 7 }}>
            <button onClick={() => setPanel(v => !v)} style={{ background: panel ? `${C.cyan}15` : `${C.cyan}08`, border: `1px solid ${C.cyan}`, color: C.cyan, borderRadius: 6, padding: "5px 12px", fontSize: 9, cursor: "pointer", fontFamily: "'JetBrains Mono', monospace" }}>
              {panel ? "▲ Fermer" : "📂 Charger données"}
            </button>
          </div>
        </div>

        {/* ── Panneau fichiers ── */}
        {panel && (
          <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8 }}>
            {[
              ["energy", parseEnergy, setEnergy, "fdqnte_energy.csv", ".csv", C.cyan, "Énergie, PDR RL, PEPM, FND/HND"],
              ["summary", parseSummary, setSummary, "fdqnte_summary.csv", ".csv", C.green, "PDR final, TX/RX, métriques"],
              ["rl", parseRL, setRl, "fdqnte_rl_history.json", ".json", C.purple, "Reward, Q-values, RL history"],
              ["topoI", parseTopo, setTopoI, "fdqnte_topology_initial.csv", ".csv", C.amber, "Topologie initiale (t=0)"],
              ["topoF", parseTopo, setTopoF, "fdqnte_topology_final.csv", ".csv", C.red, "Topologie finale (t=fin)"],
              ["routing", parseRouting, setRouting, "fdqnte_routing.csv", ".csv", C.purple, "Traces de routage RL (170k lignes)"],
            ].map(([key, parser, setter, name, accept, color, hint]) => (
              <div key={key}>
                <div style={{ color: C.dim, fontSize: 8, marginBottom: 3, letterSpacing: 1 }}>
                  <span style={{ fontFamily:"'JetBrains Mono',monospace", fontSize:7 }}>{name.replace("fdqnte_","").replace("comparison_metrics","comparison")}</span>
                </div>
                {files[key]
                  ? <Badge name={files[key]} color={color} onClear={() => { setter(null); setFiles(f => { const n = { ...f }; delete n[key]; return n; }); }} />
                  : <DropZone label={name.replace("fdqnte_","").replace("comparison_metrics","comparison").replace(/_/g," ")} accept={accept} hint={hint} color={color} onLoad={load(key, parser, setter)} />
                }
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── KPI CARDS ── */}
      <div style={{ padding: "12px 20px 0", display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 8 }}>
        <KPI label="FND" value={fndT ? `${fndT}s` : "—"} color={C.amber} sub="1er nœud mort" />
        <KPI label="HND" value={hndT ? `${hndT}s` : "—"} color={C.purple} sub="50% morts" />
        <KPI label="PDR RL" value={pdrRL ? `${pdrRL.toFixed(1)}` : "—"} unit="%" color={pdrColor} sub={`${summary?.RL_PktDelivered?.toLocaleString() || "—"}/${summary?.RL_PktEmitted?.toLocaleString() || "—"}`} />
        <KPI label="Délai E2E" value={summary?.AvgDelay_ms ? `${(+summary.AvgDelay_ms).toFixed(2)}` : last.delay ? `${last.delay.toFixed(2)}` : "—"} unit="ms" color={C.blue} sub="end-to-end" />
        <KPI label="E drainée" value={summary?.EnergyTotalConsumed_J ? `${(+summary.EnergyTotalConsumed_J).toFixed(1)}` : rl?.metrics?.total_energy_consumed_J ? `${rl.metrics.total_energy_consumed_J.toFixed(1)}` : last.drained ? `${last.drained.toFixed(1)}` : "—"} unit="J" color={C.cyan} sub={`moy ${summary?.EnergyMean_J ? (+summary.EnergyMean_J).toFixed(3) : last.energy?.toFixed(3) || "—"}J`} />
        <KPI label="PEPM@risque" value={rows.length ? (last.atRisk ?? 0) : "—"} unit={rows.length ? " nœuds" : ""} color={(last.atRisk ?? 0) > 50 ? C.red : C.green} sub="seuil 0.70" />
        <KPI label="RL Steps" value={rows.length ? (last.rlSteps||0).toLocaleString() : rl ? (rl.history?.at(-1)?.rlSteps||summary?.RL_Steps||0).toLocaleString() : "—"} color={C.purple} sub={rows.length ? `fed: ${last.fedRound||0}` : rl ? `fed: ${rl.history?.at(-1)?.fedRound||summary?.FedRounds||0}` : "—"} />
      </div>

      {/* ── TABS ── */}
      <div style={{ padding: "12px 20px 0" }}>
        <div style={{ display: "flex", gap: 0, borderBottom: `1px solid ${C.border}`, overflowX: "auto" }}>
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: tab === t.id ? C.panel : "transparent",
              border: "none",
              borderBottom: tab === t.id ? `2px solid ${C.cyan}` : "2px solid transparent",
              color: tab === t.id ? C.txt : C.dim,
              padding: "7px 14px",
              cursor: "pointer",
              fontSize: 9,
              letterSpacing: 1,
              fontFamily: "'JetBrains Mono', monospace",
              transition: "all .2s",
              whiteSpace: "nowrap"
            }}>
              {t.label}
              {files[t.id === "pdr" ? "energy" : t.id === "topo" ? "topoF" : t.id] &&
                <span style={{ display: "inline-block", width: 5, height: 5, borderRadius: "50%", background: C.green, marginLeft: 5, verticalAlign: "middle" }} />}
            </button>
          ))}
        </div>
      </div>

      {/* ── CONTENU ── */}
      <div style={{ padding: "12px 20px 24px" }}>

        {/* ═══ ÉNERGIE & DURÉE DE VIE ═══ */}
        {tab === "energy" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {!rows.length && (
              <div style={{ background: `${C.cyan}08`, border: `1px solid ${C.cyan}30`, borderRadius: 7, padding: "11px 16px", fontSize: 9, color: C.dim, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>📂</span>
                <span>Charge <code style={{ color: C.cyan }}>fdqnte_energy.csv</code> pour afficher les courbes d'énergie, FND/HND, et clusters.</span>
              </div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>

            {/* Énergie moy + min */}
            <Panel>
              <Title accent={C.cyan}>Énergie résiduelle moyenne (J)</Title>
              <ResponsiveContainer width="100%" height={210}>
                <ComposedChart data={rows}>
                  <defs>
                    <linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.cyan} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={C.cyan} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="emng" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.red} stopOpacity={0.15} />
                      <stop offset="95%" stopColor={C.red} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "t (s)", position: "insideRight", fill: C.dim, fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} domain={[0, "auto"]} />
                  <Tooltip {...tt} formatter={(v, n) => [`${v?.toFixed(4)} J`, n]} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area type="monotone" dataKey="energy" name="E moy" stroke={C.cyan} strokeWidth={2} fill="url(#eg)" />
                  <Area type="monotone" dataKey="energyMin" name="E min" stroke={C.red} strokeWidth={1} strokeDasharray="4 2" fill="url(#emng)" />
                  <Line type="monotone" dataKey="energyMax" name="E max" stroke={C.green} strokeWidth={1} strokeDasharray="3 3" dot={false} />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                  {hndT > 0 && <ReferenceLine x={hndT} stroke={C.red} strokeDasharray="5 3" label={{ value: "HND", fill: C.red, fontSize: 8, position: "top" }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>

            {/* Vivants / morts */}
            <Panel>
              <Title accent={C.green}>Nœuds vivants / morts</Title>
              <ResponsiveContainer width="100%" height={210}>
                <AreaChart data={rows}>
                  <defs>
                    <linearGradient id="ag" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.green} stopOpacity={.25} />
                      <stop offset="95%" stopColor={C.green} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="dg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.red} stopOpacity={.25} />
                      <stop offset="95%" stopColor={C.red} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} domain={[0, N + 20]} />
                  <Tooltip {...tt} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area type="monotone" dataKey="alive" name="Vivants" stroke={C.green} strokeWidth={2} fill="url(#ag)" />
                  <Area type="monotone" dataKey="dead" name="Morts" stroke={C.red} strokeWidth={2} fill="url(#dg)" />
                  <ReferenceLine y={N / 2} stroke={C.red} strokeDasharray="4 2" label={{ value: "seuil HND", fill: C.red, fontSize: 8, position: "right" }} />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                  {hndT > 0 && <ReferenceLine x={hndT} stroke={C.red} strokeDasharray="5 3" label={{ value: "HND", fill: C.red, fontSize: 8, position: "top" }} />}
                </AreaChart>
              </ResponsiveContainer>
            </Panel>

            {/* Énergie totale drainée */}
            <Panel>
              <Title accent={C.amber}>Énergie totale drainée (J)</Title>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={rows}>
                  <defs>
                    <linearGradient id="drg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.amber} stopOpacity={.3} />
                      <stop offset="95%" stopColor={C.amber} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} />
                  <Tooltip {...tt} formatter={v => [`${v?.toFixed(2)} J`, "Drainé"]} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area type="monotone" dataKey="drained" name="Drainé total" stroke={C.amber} strokeWidth={2} fill="url(#drg)" />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                </AreaChart>
              </ResponsiveContainer>
            </Panel>

            {/* Clusters actifs — NClusters depuis energy.csv */}
            <Panel>
              <Title accent={C.purple}>Clusters IFO actifs au fil du temps</Title>
              <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                Nb de CH actifs par round · re-clustering IFO adaptatif
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={rows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "t (s)", position: "insideRight", fill: C.dim, fontSize: 8 }} />
                  <YAxis yAxisId="l" stroke={C.dim} tick={{ fontSize: 8 }} domain={[0, 30]} label={{ value: "Clusters", angle: -90, position: "insideLeft", fill: C.dim, fontSize: 8, style: { textAnchor: "middle" } }} />
                  <YAxis yAxisId="r" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "Vivants", angle: 90, position: "insideRight", fill: C.dim, fontSize: 8, style: { textAnchor: "middle" } }} />
                  <Tooltip {...tt} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Bar  yAxisId="l" dataKey="nClusters" name="Clusters actifs" fill={C.purple} fillOpacity={0.7} radius={[2,2,0,0]} />
                  <Line yAxisId="r" type="monotone" dataKey="alive" name="Nœuds vivants" stroke={C.green} strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                  {fndT > 0 && <ReferenceLine yAxisId="l" x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                  {hndT > 0 && <ReferenceLine yAxisId="l" x={hndT} stroke={C.red} strokeDasharray="5 3" label={{ value: "HND", fill: C.red, fontSize: 8 }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>
          </div>
          </div>
        )}

        {/* ═══ PDR & DÉLAI ═══ */}
        {tab === "pdr" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {!rows.length && (
              <div style={{ background: `${C.cyan}08`, border: `1px solid ${C.cyan}30`, borderRadius: 7, padding: "11px 16px", fontSize: 9, color: C.dim, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>📂</span>
                <span>Charge <code style={{ color: C.cyan }}>fdqnte_energy.csv</code> pour afficher les courbes PDR et délai.</span>
              </div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>

            {/* PDR RL évolution */}
            <Panel>
              <Title accent={C.cyan}>PDR RL (%) — évolution</Title>
              <div style={{ color: C.dim, fontSize: 8, marginBottom: 8, background: `${C.cyan}08`, border: `1px solid ${C.border}`, borderRadius: 5, padding: "5px 8px", lineHeight: 1.7 }}>
                <span style={{ color: C.cyan }}>PDR RL</span> = paquets livrés selon la topologie LEACH hiérarchique.<br />
                Décroît progressivement après FND quand les CH meurent.
              </div>
              <ResponsiveContainer width="100%" height={190}>
                <LineChart data={rows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} domain={rows.length ? [Math.max(0, Math.floor(Math.min(...rows.map(r=>r.pdrRL||100))-1)), 101] : [85, 101]} />
                  <Tooltip content={<PDRTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Line type="monotone" dataKey="pdrRL" name="PDR RL" stroke={C.cyan} strokeWidth={2} dot={false} />
                  <ReferenceLine y={90} stroke={C.amber} strokeDasharray="4 2" label={{ value: "cible 90%", fill: C.amber, fontSize: 8, position: "right" }} />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                  {hndT > 0 && <ReferenceLine x={hndT} stroke={C.red} strokeDasharray="5 3" label={{ value: "HND", fill: C.red, fontSize: 8, position: "top" }} />}
                </LineChart>
              </ResponsiveContainer>
            </Panel>

            {/* Stats PDR détaillées */}
            <Panel>
              <Title accent={C.green}>Statistiques PDR détaillées</Title>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                {[
                  ["PDR RL final",     pdrRL ? `${pdrRL.toFixed(2)}%` : "—",                                                                        pdrColor],
                  ["PDR moyen sim",    rows.length ? `${(rows.reduce((s,r)=>s+(r.pdrRL||0),0)/rows.length).toFixed(2)}%` : "—",                     C.cyan],
                  ["PDR min observé",  rows.length ? `${Math.min(...rows.map(r=>r.pdrRL||100)).toFixed(2)}%` : "—",                                 C.red],
                  ["PDR max observé",  rows.length ? `${Math.max(...rows.map(r=>r.pdrRL||0)).toFixed(2)}%` : "—",                                   C.green],
                  ["Paquets émis",     summary?.RL_PktEmitted?.toLocaleString() || "—",                                                              C.txt],
                  ["Paquets livrés",   summary?.RL_PktDelivered?.toLocaleString() || "—",                                                            C.green],
                  ["Non livrés",       summary?.RL_PktEmitted && summary?.RL_PktDelivered ? (summary.RL_PktEmitted - summary.RL_PktDelivered).toLocaleString() : "—", C.red],
                  ["Délai E2E moy",    `${summary?.AvgDelay_ms?.toFixed(2) || last.delay?.toFixed(2) || "—"} ms`,                                   C.blue],
                ].map(([k,v,c]) => (
                  <div key={k} style={{ background: C.muted, borderRadius: 5, padding: "7px 9px" }}>
                    <div style={{ color: C.dim, fontSize: 8, marginBottom: 2 }}>{k}</div>
                    <div style={{ color: c, fontWeight: 700, fontSize: 13, fontFamily: "'JetBrains Mono',monospace" }}>{v}</div>
                  </div>
                ))}
              </div>
              <div style={{ background: C.muted, borderRadius: 4, height: 8, overflow: "hidden" }}>
                <div style={{ width: `${Math.min(100, pdrRL)}%`, height: "100%", borderRadius: 4, background: `linear-gradient(90deg,${C.red},${C.amber},${C.green})`, boxShadow: `0 0 8px ${pdrColor}`, transition: "width .5s" }} />
              </div>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 8, color: C.dim, marginTop: 3 }}>
                <span>0%</span><span style={{ color: pdrColor, fontWeight: 700 }}>{pdrRL.toFixed(1)}%</span><span>100%</span>
              </div>
            </Panel>

            {/* PDR RL — paquets émis vs livrés */}
            <Panel>
              <Title accent={C.purple}>Paquets émis vs livrés (cumulatif)</Title>
              <ResponsiveContainer width="100%" height={195}>
                <ComposedChart data={rows}>
                  <defs>
                    <linearGradient id="emg2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.purple} stopOpacity={.2} />
                      <stop offset="95%" stopColor={C.purple} stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="dlvg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.green} stopOpacity={.2} />
                      <stop offset="95%" stopColor={C.green} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} tickFormatter={v => `${(v / 1000).toFixed(0)}k`} />
                  <Tooltip {...tt} formatter={(v, n) => [v?.toLocaleString(), n]} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area type="monotone" dataKey="rlEmit" name="Émis" stroke={C.purple} strokeWidth={1.5} fill="url(#emg2)" />
                  <Area type="monotone" dataKey="rlDeliv" name="Livrés" stroke={C.green} strokeWidth={2} fill="url(#dlvg)" />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>

            {/* Délai E2E */}
            <Panel>
              <Title accent={C.blue}>Délai End-to-End (ms)</Title>
              <ResponsiveContainer width="100%" height={195}>
                <LineChart data={rows}>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} />
                  <Tooltip {...tt} formatter={v => [`${v?.toFixed(2)} ms`, "Délai"]} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Line type="monotone" dataKey="delay" name="Délai E2E" stroke={C.blue} strokeWidth={2} dot={false} />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                </LineChart>
              </ResponsiveContainer>
            </Panel>
          </div>
          </div>
        )}

        {/* ═══ APPRENTISSAGE RL ═══ */}
        {tab === "rl" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {!rows.length && !rlRows.length && (
              <div style={{ background: `${C.purple}08`, border: `1px solid ${C.purple}30`, borderRadius: 7, padding: "11px 16px", fontSize: 9, color: C.dim, display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 18 }}>📂</span>
                <span>Charge <code style={{ color: C.purple }}>fdqnte_energy.csv</code> et/ou <code style={{ color: C.purple }}>fdqnte_rl_history.json</code> pour voir les courbes d'apprentissage.</span>
              </div>
            )}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>

            {!files.rl && (
              <div style={{ gridColumn: "1/-1", background: `${C.purple}08`, border: `1px solid ${C.purple}30`, borderRadius: 7, padding: "9px 14px", fontSize: 9, color: C.dim }}>
                <span style={{ color: C.purple }}>ℹ</span> Charge <code style={{ color: C.purple }}>fdqnte_rl_history.json</code> pour voir les données RL réelles.
              </div>
            )}

            {/* Reward min/moy/max */}
            <Panel>
              <Title accent={C.purple}>Récompense ADDQN (min/moy/max)</Title>
              <ResponsiveContainer width="100%" height={210}>
                <ComposedChart data={rlRows.length ? rlRows : rows}>
                  <defs>
                    <linearGradient id="rwg" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={C.purple} stopOpacity={.3} />
                      <stop offset="95%" stopColor={C.purple} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "t (s)", position: "insideRight", fill: C.dim, fontSize: 8 }} />
                  <YAxis stroke={C.dim} tick={{ fontSize: 8 }} />
                  <Tooltip {...tt} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area type="monotone" dataKey="rewardMean" name="Récompense moyenne" stroke={C.purple} strokeWidth={2} fill="url(#rwg)" />
                  <Line type="monotone" dataKey="rewardMax" name="Récompense max" stroke={C.green} strokeWidth={1} strokeDasharray="4 2" dot={false} />
                  <Line type="monotone" dataKey="rewardMin" name="Récompense min" stroke={C.red} strokeWidth={1} strokeDasharray="4 2" dot={false} />
                  {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8, position: "top" }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>

            {/* RL Steps & Fed rounds */}
            <Panel>
              <Title accent={C.cyan}>Steps RL & Rounds fédérés</Title>
              <ResponsiveContainer width="100%" height={210}>
                {(() => {
                    const stepData = rlRows.length ? rlRows : rows;
                    const fndX     = fndT || (rl?.metrics?.fnd_time_s) || 0;
                    return (
                      <ComposedChart data={stepData}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value:"t(s)", position:"insideRight", fill:C.dim, fontSize:8 }}/>
                        <YAxis yAxisId="left" stroke={C.dim} tick={{ fontSize: 8 }} tickFormatter={v => `${(v/1000).toFixed(0)}k`} label={{ value:"RL Steps", angle:-90, position:"insideLeft", fill:C.dim, fontSize:8, style:{textAnchor:"middle"} }}/>
                        <YAxis yAxisId="right" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} tickFormatter={v => `${(v/1000).toFixed(1)}k`} label={{ value:"Fed Rounds", angle:90, position:"insideRight", fill:C.dim, fontSize:8, style:{textAnchor:"middle"} }}/>
                        <Tooltip {...tt} formatter={(v, n) => [v?.toLocaleString(), n]} />
                        <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                        <Line yAxisId="left"  type="monotone" dataKey="rlSteps"  name="RL Steps"    stroke={C.cyan}   strokeWidth={2}   dot={false}/>
                        <Line yAxisId="right" type="monotone" dataKey="fedRound" name="Fed Rounds"  stroke={C.purple} strokeWidth={1.5} strokeDasharray="4 2" dot={false}/>
                        {fndX > 0 && <ReferenceLine yAxisId="left" x={fndX} stroke={C.amber} strokeDasharray="5 3" label={{value:"FND",fill:C.amber,fontSize:8}}/>}
                      </ComposedChart>
                    );
                  })()}
              </ResponsiveContainer>
            </Panel>

            {/* Epsilon ε-greedy + PEPM@risque combiné */}
            <Panel>
              <Title accent={C.red}>PEPM@risque & Décroissance ε-greedy</Title>
              <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                Nœuds PEPM@risque (axe gauche) · ε décroît de 1.0→0.05 (axe droit)
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <ComposedChart data={rlRows.length ? rlRows : rows}>
                  <defs>
                    <linearGradient id="peg2" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%"  stopColor={C.red} stopOpacity={0.25}/>
                      <stop offset="95%" stopColor={C.red} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                  <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                  <YAxis yAxisId="l" stroke={C.dim} tick={{ fontSize: 8 }} domain={[0, Math.max(N, last.atRisk || 0)]} label={{ value: "@risque", angle: -90, position: "insideLeft", fill: C.dim, fontSize: 8, style: { textAnchor: "middle" } }} />
                  <YAxis yAxisId="r" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} domain={[0, 1.1]} tickFormatter={v => v.toFixed(2)} label={{ value: "ε", angle: 90, position: "insideRight", fill: C.dim, fontSize: 8, style: { textAnchor: "middle" } }} />
                  <Tooltip {...tt} />
                  <Legend wrapperStyle={{ fontSize: 8, color: C.dim, paddingTop: 10 }} />
                  <Area yAxisId="l" type="monotone" dataKey="atRisk"  name="PEPM@risque" stroke={C.red}  strokeWidth={2} fill="url(#peg2)" />
                  {fndT > 0 && <ReferenceLine yAxisId="l" x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                </ComposedChart>
              </ResponsiveContainer>
            </Panel>

            {/* Config & métriques RL */}
            <Panel>
              <Title accent={C.green}>Config ADDQN & Métriques finales</Title>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 4 }}>
                {[
                  // Paramètres fixes ADDQN (fdqn_config.h)
                  ["γ (gamma)",       "0.99",                                                                        C.cyan],
                  ["α (LR)",          "3×10⁻⁴",                                                                     C.cyan],
                  ["ε_min",           "0.05",                                                                        C.green],
                  ["ε_decay",         "0.995 / step",                                                               C.green],
                  ["Fed period",      "50 rounds",                                                                   C.purple],
                  ["Batch size",      "64",                                                                          C.purple],
                  // Métriques dynamiques depuis les fichiers chargés
                  ["RL Steps total",  rows.length ? (last.rlSteps || 0).toLocaleString() : (rl?.history?.at(-1)?.rlSteps || 0).toLocaleString() || "—",  C.cyan],
                  ["Fed Rounds",      rows.length ? (last.fedRound || 0).toLocaleString() : (rl?.history?.at(-1)?.fedRound || 0).toLocaleString() || "—", C.purple],
                  ["IFO Rounds",      rows.length ? (last.ifoRound || 0).toString() : (summary?.IFO_Rounds || "—").toString(),                            C.amber],
                  ["N nœuds",         (rl?.info?.nNodes || summary?.N || N).toString(),                              C.txt],
                  ["Zone",            rl?.info?.areaSize_m ? `${rl.info.areaSize_m}×${rl.info.areaSize_m}m` : (summary?.AreaSize_m ? `${summary.AreaSize_m}m` : "—"), C.txt],
                  ["Radio range",     rl?.info?.radioRange_m ? `${rl.info.radioRange_m}m` : (summary?.RadioRange_m ? `${summary.RadioRange_m}m` : "—"),   C.txt],
                  ["E initiale",      rl?.info?.initEnergy_J ? `${rl.info.initEnergy_J}J` : (summary?.InitEnergy_J ? `${summary.InitEnergy_J}J` : "—"),  C.amber],
                  ["Sim durée",       rl?.info?.simDuration_s ? `${rl.info.simDuration_s}s` : (summary?.SimDuration_s ? `${summary.SimDuration_s}s` : "—"), C.txt],
                  ["Seed",            (rl?.info?.seed || summary?.Seed || "—").toString(),                           C.dim],
                ].map(([k, v, c]) => (
                  <div key={k} style={{ background: C.muted, borderRadius: 4, padding: "5px 8px", display: "flex", justifyContent: "space-between", gap: 8, fontSize: 9, alignItems: "center" }}>
                    <span style={{ color: C.dim }}>{k}</span>
                    <span style={{ color: c, fontFamily: "'JetBrains Mono', monospace", fontWeight: 700, fontSize: 10 }}>{v}</span>
                  </div>
                ))}
              </div>
            </Panel>
          </div>
          </div>
        )}

        {/* ═══ TOPOLOGIE ═══ */}
        {tab === "topo" && (() => {
          // Variables locales topologie
          const topoAlive  = topo.filter(n=>n.isAlive);
          const topo_ch    = topoAlive.filter(n=>n.isCH);
          const atRiskT    = topoAlive.filter(n=>n.pepmRisk>0.7);
          const topoDead   = topo.filter(n=>!n.isAlive);
          const meanEN     = topoAlive.length?(topoAlive.reduce((a,n)=>a+(n.energyNorm??0),0)/topoAlive.length).toFixed(3):"—";
          const stdE       = last.energyStd??0;
          const pdrDisp    = summary?.PDR_RL_pct!=null?`${(+summary.PDR_RL_pct).toFixed(1)}%`:(files.topoF||files.topoI)?"—":"—";
          const epsilon    = rl?.history?.length?(Math.max(0.05,Math.pow(0.995,rl.history.at(-1)?.rlSteps||0))).toFixed(3):"—";
          const rlSteps    = last.rlSteps || rl?.history?.at(-1)?.rlSteps || 0;

          const metrics=[
            {label:"Vivants",      val:`${topoAlive.length}/${topo.length}`,
             sub:`${topoDead.length} morts`,   col:C.green,
             pct:topoAlive.length/Math.max(topo.length,1)*100},
            {label:"CH actifs",    val:`${topo_ch.length}`,
             sub:"cluster heads",              col:C.cyan,
             pct:topo_ch.length/Math.max(topoAlive.length,1)*100*4},
            {label:"E moy vivants",val:meanEN+"J",
             sub:"énergie résiduelle",         col:"#ff8c00",
             pct:parseFloat(meanEN)/2*100},
            {label:"PEPM@risque",  val:`${atRiskT.length}`,
             sub:"seuil > 70%",               col:atRiskT.length>0?C.red:C.green,
             pct:atRiskT.length/Math.max(topoAlive.length,1)*100},
            {label:"Moy dist Sink",val:topoAlive.length?`${(topoAlive.reduce((s,n)=>s+(n.distToSink??0),0)/topoAlive.length).toFixed(0)}m`:"—",
             sub:"distance au sink",          col:C.purple,
             pct:null},
            {label:"Fitness moy",  val:topo_ch.length?`${(topo_ch.reduce((s,n)=>s+(n.fitness??0),0)/topo_ch.length).toFixed(3)}`:"—",
             sub:"IFO clusters",              col:C.purple,
             pct:topo_ch.length?(topo_ch.reduce((s,n)=>s+(n.fitness??0),0)/topo_ch.length)/1.5*100:0},
          ];

          return (
<div style={{display:"grid",gridTemplateColumns:"1fr 290px",gap:12,
      height:"calc(100vh - 215px)",minHeight:540}}>

      {/* ── CANVAS ── */}
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:12,
        display:"flex",flexDirection:"column"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
          <Title accent={C.cyan}>Topologie réseau — {files.topoF||files.topoI||"données manquantes"}</Title>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            {["initial","final"].map(m=>(
              <button key={m} onClick={()=>setTopoMode(m)} style={{
                background:topoMode===m?`${C.cyan}20`:C.muted,
                border:`1px solid ${topoMode===m?C.cyan:C.border}`,
                color:topoMode===m?C.cyan:C.dim,
                borderRadius:4,padding:"3px 10px",fontSize:8,cursor:"pointer",
                fontFamily:"'JetBrains Mono',monospace"
              }}>
                {m==="initial"?"t = 0":"t = fin"}
              </button>
            ))}
            {!(files.topoF||files.topoI)&&<span style={{color:C.amber,fontSize:8,border:`1px solid ${C.amber}40`,
              borderRadius:4,padding:"2px 7px"}}>démo · charger CSV via 📂</span>}
          </div>
        </div>
        <div style={{flex:1,minHeight:0}}>
          <TopoCanvas
            nodes={topo}
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
        <div style={{display:"flex",gap:12,flexWrap:"wrap",marginTop:6,fontSize:9,color:C.dim}}>
          {[[clCol(2),"Cluster Head"],["#2a3e58","Membre"],["#ff3860","Mort"],["#ffd700","Sink/BS"],
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

        <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:8,padding:10}}>
          <Title accent={C.amber}>Statistiques topologie</Title>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
            {metrics.map(m=>(
              <div key={m.label} style={{background:"#0f1e30",border:`1px solid ${C.border}`,
                borderRadius:6,padding:"7px 9px"}}>
                <div style={{fontSize:9,fontWeight:700,textTransform:"uppercase",letterSpacing:.8,
                  color:C.dim,marginBottom:2}}>{m.label}</div>
                <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:15,fontWeight:700,
                  color:m.col,lineHeight:1}}>{m.val}</div>
                <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:9,color:C.dim,marginTop:1}}>{m.sub}</div>
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
            <div style={{color:C.dim,fontSize:10,textAlign:"center",padding:"12px 0",lineHeight:2.2}}>
              ⊕<br/>Cliquez un nœud<br/>sur la carte
            </div>
          ):(()=>{
            const rc=selNode.pepmRisk>.7?C.red:selNode.pepmRisk>.4?"#ff8c00":C.green;
            const nodeClCol=clCol(selNode.clusterId);
            return(
              <div>
                <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:7,
                  background:"#0f1e30",borderRadius:5,padding:"6px 9px"}}>
                  <div style={{width:9,height:9,borderRadius:"50%",background:nodeClCol,
                    boxShadow:`0 0 5px ${nodeClCol}`}}/>
                  <span style={{color:C.txt,fontWeight:700,fontFamily:"'JetBrains Mono',monospace",fontSize:12}}>
                    #{selNode.id}{selNode.isCH?" ⬡":""}
                  </span>
                  <span style={{color:selNode.isAlive?C.green:C.red,fontSize:10,marginLeft:"auto"}}>
                    {selNode.isAlive?"✓ Vivant":"✗ Mort"}
                  </span>
                </div>
                {[
                  ["Node ID",     `#${selNode.id}`,                                             C.txt],
                  ["Position",    `(${selNode.x}m, ${selNode.y}m)`,                             C.txt],
                  ["Cluster",     `C${selNode.clusterId}${selNode.isCH?" · CH":""}`,            nodeClCol],
                  ["Énergie",     `${(selNode.energy??0).toFixed(4)} J`,                        "#ff8c00"],
                  ["Risque PEPM", `${((selNode.pepmRisk??0)*100).toFixed(1)}%`,                 rc],
                  ["Dist. Sink",  `${(selNode.distToSink??0).toFixed(1)} m`,                    C.txt],
                  ["Voisins",     `${selNode.nbCount??0} (${150}m)`,                            C.txt],
                  ["Fitness IFO", `${(selNode.fitness??0).toFixed(4)}`,                         C.purple],
                ].map(([k,v,c])=>(
                  <div key={k} style={{display:"flex",justifyContent:"space-between",
                    padding:"3px 0",borderBottom:`1px solid rgba(255,255,255,.04)`,fontSize:10}}>
                    <span style={{color:C.dim,fontFamily:"'JetBrains Mono',monospace"}}>{k}</span>
                    <span style={{color:c,fontFamily:"'JetBrains Mono',monospace"}}>{v}</span>
                  </div>
                ))}
                <div style={{marginTop:7}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.dim,marginBottom:2}}>
                    <span>Énergie</span><span style={{color:"#ff8c00"}}>{((selNode.energyNorm??0)*100).toFixed(1)}%</span>
                  </div>
                  <div style={{background:C.muted,borderRadius:3,height:4,overflow:"hidden"}}>
                    <div style={{width:`${(selNode.energyNorm??0)*100}%`,height:"100%",borderRadius:3,
                      background:`linear-gradient(90deg,${C.red},#ff8c00,#00ff88)`,transition:"width .3s"}}/>
                  </div>
                </div>
                <div style={{marginTop:5}}>
                  <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:C.dim,marginBottom:2}}>
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
            <div style={{color:C.dim,fontSize:10,lineHeight:1.8}}>
              Sélectionnez un nœud pour tracer son chemin greedy vers le Sink (500,500).
            </div>
          ):(
            <div>
              {/* chemin avec coordonnées, identique au HTML */}
              <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:9,lineHeight:2,
                wordBreak:"break-all",maxHeight:120,overflowY:"auto"}}>
                {selPath.ids.map((id,i)=>{
                  const n=topo.find(n=>n.id===id);
                  const isSrc=i===0,isCH=n?.isCH;
                  const col=isSrc||isCH?C.green:C.cyan;
                  return(
                    <span key={i}>
                      <span style={{color:col}}>#{id}({n?.x},{n?.y})</span>
                      {i<selPath.ids.length-1&&<span style={{color:C.dim}}> →{"\n"}</span>}
                    </span>
                  );
                })}
                <span style={{color:C.dim}}> → </span>
                <span style={{color:"#ffd700"}}>SINK(500,500)</span>
              </div>
              <div style={{marginTop:5,display:"flex",gap:10,fontSize:9,alignItems:"center"}}>
                <span style={{color:C.dim}}>{selPath.ids.length} hops</span>
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
          <div style={{fontSize:9,color:C.dim,marginBottom:6,lineHeight:1.7,fontFamily:"'JetBrains Mono',monospace"}}>
            F = W1·E + W2·dSink + W3·deg − W4·rot
          </div>
          {[["W1 énergie","0.45",C.green],["W2 proximité","0.25",C.cyan],
            ["W3 densité","0.20",C.purple],["W4 rotation","0.10","#ff8c00"]].map(([l,v,c])=>(
            <div key={l} style={{marginBottom:5}}>
              <div style={{display:"flex",justifyContent:"space-between",fontSize:9,marginBottom:2}}>
                <span style={{color:C.dim}}>{l}</span><span style={{color:c}}>{v}</span>
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
        })()}


        {/* ═══ ROUTAGE RL ═══ */}
        {tab === "routing" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

            {/* Bandeau chargement */}
            {!routing ? (
              <div style={{ background: `${C.purple}08`, border: `1px solid ${C.purple}30`,
                borderRadius: 7, padding: "18px 20px", textAlign: "center", color: C.dim, fontSize: 10 }}>
                <div style={{ fontSize: 24, marginBottom: 8 }}>📡</div>
                <div>Charge <code style={{ color: C.purple }}>fdqnte_routing.csv</code> via <strong style={{ color: C.txt }}>📂 Charger données</strong></div>
                <div style={{ fontSize: 9, marginTop: 4, color: C.dim }}>170 000+ traces de routage — PDR, drain énergie, PEPM, CH actifs</div>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

                {/* KPIs routing */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(6,1fr)", gap: 8 }}>
                  {[
                    ["Total paquets", routing.summary.totalPackets.toLocaleString(), C.cyan,   "traces"],
                    ["Livrés",        routing.summary.delivered.toLocaleString(),    C.green,  `${routing.summary.pdrGlobal}%`],
                    ["Non livrés",    routing.summary.nonDelivered,                  routing.summary.nonDelivered > 0 ? C.red : C.green, "pertes"],
                    ["PDR global",    `${routing.summary.pdrGlobal}%`,               routing.summary.pdrGlobal > 99 ? C.green : C.amber, "logique"],
                    ["E drain moy",   `${(routing.summary.drainMoy*1000).toFixed(2)}mJ`, C.amber, "par nœud total"],
                    ["Nœuds actifs",  routing.summary.nNodes,                        C.purple, `${routing.summary.nCHActive} CH`],
                  ].map(([l, v, c, sub]) => (
                    <div key={l} style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8,
                      padding: "10px 13px", position: "relative", overflow: "hidden" }}>
                      <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 2, background: c }} />
                      <div style={{ color: C.dim, fontSize: 8, letterSpacing: 2, textTransform: "uppercase", marginBottom: 3 }}>{l}</div>
                      <div style={{ color: c, fontSize: 18, fontWeight: 700, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1 }}>{v}</div>
                      <div style={{ color: C.dim, fontSize: 8, marginTop: 2 }}>{sub}</div>
                    </div>
                  ))}
                </div>

                {/* Ligne 1: PDR temporel + Paquets/drain */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Panel>
                    <Title accent={C.cyan}>PDR logique RL (%) — évolution temporelle</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      Taux de livraison par round · <span style={{ color: C.green }}>100% jusqu'à FND</span> puis décroît avec les morts de CH
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <ComposedChart data={routing.timeSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "t (s)", position: "insideRight", fill: C.dim, fontSize: 8 }} />
                        <YAxis yAxisId="l" stroke={C.dim} tick={{ fontSize: 8 }} domain={[98, 101]} />
                        <YAxis yAxisId="r" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} />
                        <Tooltip {...tt} formatter={(v, n) => [typeof v === "number" ? (n === "PDR%" ? `${v.toFixed(2)}%` : v.toLocaleString()) : v, n]} />
                        <Legend wrapperStyle={{ fontSize: 8, color: C.dim }} />
                        <Line yAxisId="l" type="monotone" dataKey="pdr"     name="PDR%"     stroke={C.cyan}  strokeWidth={2} dot={false} />
                        <Area yAxisId="r" type="monotone" dataKey="packets" name="Paquets"  stroke={C.green} strokeWidth={1} fill={`${C.green}10`} dot={false} />
                        {fndT > 0 && <ReferenceLine yAxisId="l" x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </Panel>

                  <Panel>
                    <Title accent={C.amber}>Énergie drainée par round (mJ total réseau)</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      Drain cumulatif de tous les nœuds · décroît après FND car moins de nœuds actifs
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <ComposedChart data={routing.timeSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                        <YAxis yAxisId="l" stroke={C.dim} tick={{ fontSize: 8 }} tickFormatter={v => `${(v/1000).toFixed(1)}J`} />
                        <YAxis yAxisId="r" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} />
                        <Tooltip {...tt} formatter={(v, n) => [n === "Drain (mJ)" ? `${v.toFixed(1)} mJ` : v, n]} />
                        <Legend wrapperStyle={{ fontSize: 8, color: C.dim }} />
                        <Area yAxisId="l" type="monotone" dataKey="drain_mJ" name="Drain (mJ)" stroke={C.amber} strokeWidth={2} fill={`${C.amber}20`} />
                        <Line yAxisId="r" type="monotone" dataKey="nSrc"     name="Nœuds actifs" stroke={C.purple} strokeWidth={1.5} strokeDasharray="4 2" dot={false} />
                        {fndT > 0 && <ReferenceLine yAxisId="l" x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>

                {/* Ligne 2: CH actifs + PEPM */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Panel>
                    <Title accent={C.purple}>CH actifs & Nœuds sources par round</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      Corrélation entre mort des CH et réduction du trafic
                    </div>
                    <ResponsiveContainer width="100%" height={195}>
                      <ComposedChart data={routing.timeSeries}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                        <YAxis yAxisId="l" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "CH", angle: -90, position: "insideLeft", fill: C.dim, fontSize: 8 }} domain={[0, 30]} />
                        <YAxis yAxisId="r" orientation="right" stroke={C.dim} tick={{ fontSize: 8 }} label={{ value: "Sources", angle: 90, position: "insideRight", fill: C.dim, fontSize: 8 }} />
                        <Tooltip {...tt} />
                        <Legend wrapperStyle={{ fontSize: 8, color: C.dim }} />
                        <Bar  yAxisId="l" dataKey="nCH"  name="CH actifs"     fill={C.cyan}   fillOpacity={0.7} radius={[2,2,0,0]} />
                        <Line yAxisId="r" dataKey="nSrc"  name="Nœuds sources" stroke={C.green} strokeWidth={2} dot={false} />
                        {fndT > 0 && <ReferenceLine yAxisId="l" x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </Panel>

                  <Panel>
                    <Title accent={C.red}>Risque PEPM moyen (routage) — évolution</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      PEPM risk moyen des nœuds en transmission · monte progressivement avec l'épuisement
                    </div>
                    <ResponsiveContainer width="100%" height={195}>
                      <AreaChart data={routing.timeSeries}>
                        <defs>
                          <linearGradient id="prg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%"  stopColor={C.red} stopOpacity={0.35} />
                            <stop offset="95%" stopColor={C.red} stopOpacity={0}    />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} />
                        <XAxis dataKey="time" stroke={C.dim} tick={{ fontSize: 8 }} />
                        <YAxis stroke={C.dim} tick={{ fontSize: 8 }} domain={[0.2, 0.75]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                        <Tooltip {...tt} formatter={v => [`${(v*100).toFixed(1)}%`, "PEPM risk moy"]} />
                        <Area type="monotone" dataKey="pepm" name="PEPM moy" stroke={C.red} strokeWidth={2} fill="url(#prg)" />
                        <ReferenceLine y={0.7} stroke={C.red} strokeDasharray="4 2" label={{ value: "seuil 70%", fill: C.red, fontSize: 8 }} />
                        {fndT > 0 && <ReferenceLine x={fndT} stroke={C.amber} strokeDasharray="5 3" label={{ value: "FND", fill: C.amber, fontSize: 8 }} />}
                      </AreaChart>
                    </ResponsiveContainer>
                  </Panel>
                </div>

                {/* Ligne 3: Top nœuds drain + Répartition drain */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <Panel>
                    <Title accent={C.amber}>Top 10 nœuds — drain énergétique total (J)</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      Les CH consomment plus que les membres · drain max = {routing.summary.drainMax.toFixed(4)}J
                    </div>
                    <ResponsiveContainer width="100%" height={210}>
                      <BarChart data={routing.top10Drain} layout="vertical" margin={{ top: 2, right: 50, left: 35, bottom: 2 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke={C.border} horizontal={false} />
                        <XAxis type="number" stroke={C.dim} tick={{ fontSize: 8 }} tickFormatter={v => `${v.toFixed(2)}J`} />
                        <YAxis type="category" dataKey="id" stroke={C.dim} tick={{ fontSize: 8 }} width={35} tickFormatter={v => `#${v}`} />
                        <Tooltip {...tt} formatter={(v, n) => [n === "Drain (J)" ? `${v.toFixed(4)} J` : v, n]} />
                        <Bar dataKey="drain" name="Drain (J)" radius={[0, 3, 3, 0]}>
                          {routing.top10Drain.map((n, i) => (
                            <Cell key={i} fill={n.isCH ? C.cyan : C.green} fillOpacity={0.85} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                    <div style={{ display: "flex", gap: 10, fontSize: 8, color: C.dim, marginTop: 4 }}>
                      <span><span style={{ color: C.cyan }}>■</span> Cluster Head</span>
                      <span><span style={{ color: C.green }}>■</span> Membre</span>
                    </div>
                  </Panel>

                  <Panel>
                    <Title accent={C.green}>Distribution drain (J) — tous nœuds</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 6 }}>
                      Histogramme · moy={routing.summary.drainMoy.toFixed(3)}J · min={routing.summary.drainMin.toFixed(3)}J · max={routing.summary.drainMax.toFixed(3)}J
                    </div>
                    {(() => {
                      const bins = [];
                      const mn = routing.summary.drainMin, mx = routing.summary.drainMax;
                      const bw = (mx - mn) / 10;
                      for (let i = 0; i < 10; i++) {
                        const lo = mn + i * bw, hi = mn + (i+1) * bw;
                        bins.push({
                          range: `${lo.toFixed(2)}-${hi.toFixed(2)}`,
                          count: routing.nodeList.filter(n => n.drain >= lo && n.drain < hi).length,
                          lo,
                        });
                      }
                      return (
                        <ResponsiveContainer width="100%" height={210}>
                          <BarChart data={bins} margin={{ top: 2, right: 10, left: -15, bottom: 32 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke={C.border} vertical={false} />
                            <XAxis dataKey="range" stroke={C.dim} tick={{ fontSize: 7 }} angle={-40} textAnchor="end" interval={0} height={55} />
                            <YAxis stroke={C.dim} tick={{ fontSize: 8 }} />
                            <Tooltip {...tt} formatter={v => [`${v} nœuds`, "Count"]} />
                            <Bar dataKey="count" name="Nœuds" radius={[3,3,0,0]}>
                              {bins.map((b, i) => (
                                <Cell key={i} fill={b.lo > routing.summary.drainMoy ? C.amber : C.green} fillOpacity={0.8} />
                              ))}
                            </Bar>
                            <ReferenceLine x={`${routing.summary.drainMoy.toFixed(2)}-${(routing.summary.drainMoy+bw).toFixed(2)}`}
                              stroke={C.amber} strokeDasharray="4 2" label={{ value: "moy", fill: C.amber, fontSize: 7 }} />
                          </BarChart>
                        </ResponsiveContainer>
                      );
                    })()}
                  </Panel>
                </div>

                {/* Paquets non livrés */}
                {routing.nonDelivered.length > 0 && (
                  <Panel>
                    <Title accent={C.red}>Paquets non livrés ({routing.summary.nonDelivered})</Title>
                    <div style={{ fontSize: 8, color: C.dim, marginBottom: 8 }}>
                      Apparaissent principalement autour du FND quand les CH s'épuisent
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill,minmax(220px,1fr))", gap: 6, maxHeight: 180, overflowY: "auto" }}>
                      {routing.nonDelivered.map((p, i) => (
                        <div key={i} style={{ background: `${C.red}10`, border: `1px solid ${C.red}30`, borderRadius: 5, padding: "5px 9px", fontSize: 8 }}>
                          <span style={{ color: C.red }}>✗</span>
                          <span style={{ color: C.dim, marginLeft: 5 }}>t={p.time}s</span>
                          <span style={{ color: C.txt, marginLeft: 5 }}>src=#{p.src}</span>
                          <span style={{ color: C.amber, marginLeft: 5 }}>C{p.cluster}</span>
                          <span style={{ color: C.red, marginLeft: 5 }}>PEPM={+(p.pepm*100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </Panel>
                )}
              </div>
            )}
          </div>
        )}


      </div>
    </div>
  );
}
