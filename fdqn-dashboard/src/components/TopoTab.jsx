// src/components/TopoTab.jsx
import { useState } from "react";
import { C } from '../styles/colors';
import { Title } from './Title';
import { TopoCanvas } from './TopoCanvas';
import { clusterColor } from '../utils/clusterColors';

export function TopoTab({topoData,topoFile,energyData,rlData,summary}){
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