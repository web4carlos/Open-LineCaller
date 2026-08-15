import {OfficiatingEvent} from "./liveApi";
import {PresentationMode,shouldShowDecision} from "./presentationMode";
import {useOutCallPresentation} from "./useOutCallPresentation";
export default function RoleAwareCallOverlay({mode,activeEvent}:{mode:PresentationMode,activeEvent:OfficiatingEvent|null}){
 const out=useOutCallPresentation(activeEvent,2800);
 if(mode==="PLAYER"){
  if(!out)return null;
  return <div className="playerOutOverlay"><div className="playerOutWord">OUT</div><div className="playerOutSub">CALL CONFIRMED · {Math.round(out.decision_confidence*100)}%</div></div>;
 }
 if(!activeEvent||!shouldShowDecision(mode,activeEvent.final_decision))return null;
 return <div className={"refereeCallOverlay "+activeEvent.final_decision.toLowerCase()}>
  <div className="refereeCallTop"><strong>{activeEvent.final_decision}</strong><span>{Math.round(activeEvent.decision_confidence*100)}%</span></div>
  <div className="refereeCallEvidence"><span>{activeEvent.nearest_line}</span><span>{Math.abs(activeEvent.signed_distance_in).toFixed(1)} IN {activeEvent.signed_distance_in<0?"OUT":"IN"}</span><span>{activeEvent.gate_decision}</span></div>
 </div>;
}
