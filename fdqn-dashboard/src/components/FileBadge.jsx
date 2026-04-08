// src/components/FileBadge.jsx
import { C } from '../styles/colors';

export function FileBadge({name, color=C.green, onClear}){
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