// src/components/Card.jsx
import { C } from '../styles/colors';

export function Card({label, value, unit="", color=C.cyan, sub=""}){
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