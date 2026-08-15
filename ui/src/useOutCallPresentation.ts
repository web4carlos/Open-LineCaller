import {useEffect,useRef,useState} from "react";
import {OfficiatingEvent} from "./liveApi";
export function useOutCallPresentation(activeEvent:OfficiatingEvent|null,dwellMs=2800){
 const [visible,setVisible]=useState<OfficiatingEvent|null>(null);
 const timer=useRef<number|null>(null),last=useRef<number|null>(null);
 useEffect(()=>{
  if(!activeEvent||activeEvent.final_decision!=="OUT"||last.current===activeEvent.frame)return;
  last.current=activeEvent.frame;setVisible(activeEvent);
  if(timer.current!==null)window.clearTimeout(timer.current);
  timer.current=window.setTimeout(()=>{setVisible(null);timer.current=null},dwellMs);
 },[activeEvent,dwellMs]);
 useEffect(()=>()=>{if(timer.current!==null)window.clearTimeout(timer.current)},[]);
 return visible;
}
