// src/components/DropZone.jsx
import { useState, useRef } from "react";
import { C } from '../styles/colors';

export function DropZone({label, accept, onLoad, hint}){
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