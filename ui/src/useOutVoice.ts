import {useEffect,useRef} from "react";
import {OfficiatingEvent} from "./liveApi";
import {PresentationMode,shouldSpeakDecision} from "./presentationMode";
export function useOutVoice(mode:PresentationMode,e:OfficiatingEvent|null){
 const last=useRef<number|null>(null);
 useEffect(()=>{
  if(!e||!shouldSpeakDecision(mode,e.final_decision)||e.final_decision!=="OUT"||last.current===e.frame)return;
  last.current=e.frame;
  if(!("speechSynthesis" in window))return;
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance("OUT");u.rate=1.05;u.pitch=.95;u.volume=1;
  window.speechSynthesis.speak(u);
 },[mode,e]);
}
