import {useEffect,useState} from "react";
import {fetchLive,LiveFacility} from "./liveApi";
export function useLiveFacility(intervalMs=750){
 const [data,setData]=useState<LiveFacility|null>(null);
 const [connected,setConnected]=useState(false);
 useEffect(()=>{
  let alive=true;
  async function tick(){
   try{const d=await fetchLive();if(alive){setData(d);setConnected(true)}}catch{if(alive)setConnected(false)}
  }
  tick();const id=setInterval(tick,intervalMs);
  return()=>{alive=false;clearInterval(id)}
 },[intervalMs]);
 return {data,connected};
}
