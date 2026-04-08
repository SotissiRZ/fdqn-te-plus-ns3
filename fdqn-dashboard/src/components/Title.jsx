// src/components/Title.jsx
import { C } from '../styles/colors';

export function Title({children,accent=C.cyan}){
  return(
    <div style={{display:"flex",alignItems:"center",gap:9,marginBottom:12}}>
      <div style={{width:3,height:15,background:accent,borderRadius:2,boxShadow:`0 0 6px ${accent}`}}/>
      <span style={{color:C.txt,fontSize:11,letterSpacing:2,textTransform:"uppercase",fontFamily:"'Space Mono',monospace"}}>{children}</span>
    </div>
  );
}