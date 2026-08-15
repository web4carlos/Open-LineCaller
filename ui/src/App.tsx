import {useEffect,useMemo,useState} from "react";
import "./styles.css";
import {LiveCourt,OfficiatingEvent} from "./liveApi";
import {useLiveFacility} from "./useLiveFacility";
import LiveCourtVideo from "./LiveCourtVideo";
import RoleAwareCallOverlay from "./RoleAwareCallOverlay";
import {DEFAULT_PRESENTATION_MODE,PresentationMode} from "./presentationMode";
import {useOutVoice} from "./useOutVoice";

function Mark(){return <div className="mark"><span className="markBall">&#9679;</span><div><b>OPEN</b><strong>LINECALLER</strong></div></div>}
function Status({s}:{s:string}){return <span className={"status "+s.toLowerCase()}><i/>{s}</span>}
function courtNumber(id:string,index:number){
 const m=id.match(/(\d+)$/); return m?m[1].padStart(2,"0"):String(index+1).padStart(2,"0");
}
function pct(v:number|null){return v==null?null:Math.round(v*100)}

export default function App(){
 const [presentationMode,setPresentationMode]=useState<PresentationMode>(DEFAULT_PRESENTATION_MODE);
 const {data,connected}=useLiveFacility(750);
 const [selectedId,setSelectedId]=useState<string|null>(null);
 const courts=data?.courts??[];
 const selected=useMemo(()=>courts.find(c=>c.court_id===selectedId)??null,[courts,selectedId]);

 useEffect(()=>{
   if(selectedId && data && !selected) setSelectedId(null);
 },[data,selected,selectedId]);

 if(selected) return <Live court={selected} connected={connected} presentationMode={presentationMode} setPresentationMode={setPresentationMode} back={()=>setSelectedId(null)}/>;

 const active=courts.filter(c=>c.status==="LIVE").length;
 const review=courts.filter(c=>c.status==="REVIEW").length;
 const ready=courts.filter(c=>c.status==="READY").length;

 return <div className="app">
  <header><Mark/><div className="headerRight"><div className="modeSwitcher"><button className={presentationMode==="PLAYER"?"active":""} onClick={()=>setPresentationMode("PLAYER")}>PLAYER</button><button className={presentationMode==="REFEREE"?"active":""} onClick={()=>setPresentationMode("REFEREE")}>REFEREE</button></div><span className="eyebrow">{data?.facility_name??"FACILITY CONTROL"}</span><span className={"bridgeBadge "+(connected?"connected":"disconnected")}><i/>{connected?"PYTHON LIVE":"BRIDGE OFFLINE"}</span><div className="avatar">LC</div></div></header>
  <main>
   <section className="hero"><div><span className="eyebrow">OPEN LINECALLER / COMMAND</span><h1>Every court.<br/><em>One clear call.</em></h1><p>{connected?"Live facility state is now driven by the Python bridge.":"Waiting for the Open LineCaller Python bridge on port 8765."}</p></div>
    <div className="summary"><div><b>{String(courts.length).padStart(2,"0")}</b><span>COURTS</span></div><div><b>{String(active).padStart(2,"0")}</b><span>ACTIVE</span></div><div><b>{String(review).padStart(2,"0")}</b><span>REVIEW</span></div><div><b>{String(ready).padStart(2,"0")}</b><span>READY</span></div></div>
   </section>

   {!connected && <div className="offlineBanner"><b>LIVE BRIDGE OFFLINE</b><span>Start: .\.venv\Scripts\python.exe -m tools.run_live_bridge</span></div>}

   <div className="sectionTitle"><span>COURT GRID</span><small>{data?`LIVE SEQUENCE ${data.sequence}`:"WAITING FOR PYTHON"}</small></div>
   <section className="grid">
    {courts.map((c,i)=><article key={c.court_id} className={"court "+c.status.toLowerCase()} onClick={()=>setSelectedId(c.court_id)}>
      <div className="courtTop"><div><small>COURT {courtNumber(c.court_id,i)}</small><h2>{c.name}</h2></div><Status s={c.status}/></div>
      <div className="courtVisual"><div className="miniCourt"><span/><span/><span/></div><div className={"call "+c.last_call.toLowerCase()}>{c.last_call||c.status}</div></div>
      <div className="courtMeta"><div><small>MODE</small><b>{c.mode}</b></div><div><small>SCORE</small><b>{c.score||"-"}</b></div></div>
      <div className="teams"><span>{c.session_id||"NO ACTIVE SESSION"}</span><i>{pct(c.confidence)!=null?`${pct(c.confidence)}%`:"-"}</i><span>{c.camera_id||"NO CAMERA"}</span></div>
      <button>ENTER COURT <span>&#8599;</span></button>
    </article>)}
    {connected && courts.length===0 && <div className="emptyState">NO COURTS PUBLISHED BY PYTHON</div>}
   </section>
  </main>
  <footer><span><i className={connected?"pulse":"offlineDot"}/> {connected?"SYSTEM ONLINE":"BRIDGE DISCONNECTED"}</span><span>CP-0034.2 &middot; LIVE PYTHON STATE &middot; SEQ {data?.sequence??0}</span></footer>
 </div>
}

function Live({court,connected,presentationMode,setPresentationMode,back}:{court:LiveCourt,connected:boolean,presentationMode:PresentationMode,setPresentationMode:(m:PresentationMode)=>void,back:()=>void}){
 const [activeEvent,setActiveEvent]=useState<OfficiatingEvent|null>(null);
 useOutVoice(presentationMode,activeEvent);
 const confidence=pct(activeEvent?.decision_confidence??court.confidence);
 const distance=court.distance_in==null?"-":`${court.distance_in.toFixed(1)} IN`;
 return <div className="app liveApp">
  <header><Mark/><button className="back" onClick={back}>&#8592; FACILITY</button><div className="headerRight"><div className="modeSwitcher"><button className={presentationMode==="PLAYER"?"active":""} onClick={()=>setPresentationMode("PLAYER")}>PLAYER</button><button className={presentationMode==="REFEREE"?"active":""} onClick={()=>setPresentationMode("REFEREE")}>REFEREE</button></div><span className={"bridgeBadge "+(connected?"connected":"disconnected")}><i/>{connected?"PYTHON LIVE":"BRIDGE OFFLINE"}</span><Status s={court.status}/><span className="clock">{court.court_id.toUpperCase()}</span></div></header>
  <main className="liveMain">
   <div className="liveTitle"><div><span className="eyebrow">LIVE REFEREE / PYTHON STATE</span><h1>{court.name}</h1></div><div className="scoreHero"><small>LIVE SCORE</small><b>{court.score||"-"}</b></div></div>
   <section className="liveLayout">
    <div className="videoRoleWrap"><LiveCourtVideo court={court} onEventChange={(e)=>setActiveEvent(e)}/><RoleAwareCallOverlay mode={presentationMode} activeEvent={activeEvent}/></div><aside>
      <div className={"decision "+(court.last_call||court.status).toLowerCase()}><small>LAST CALL</small><strong>{activeEvent?.final_decision||court.last_call||court.status}</strong><span>{confidence!=null?`${confidence}% CONFIDENCE`:"SYSTEM STANDBY"}</span></div>
      <div className="panel"><small>OFFICIATING</small><div className="metric"><span>Nearest line</span><b>{activeEvent?.nearest_line||court.nearest_line||"-"}</b></div><div className="metric"><span>Distance</span><b>{distance}</b></div><div className="metric"><span>Evidence</span><b>{court.evidence||"-"}</b></div></div>
      <div className="panel"><small>SERVICE</small><div className="server"><b>{court.serving_team||"-"}</b><span>{court.server_number?`SERVER #${court.server_number}`:"SERVER -"} &middot; {court.service_court||"-"}</span></div></div>
    </aside>
   </section>
   <section className="controls"><button className="primary">&#9654; START / RESUME</button><button>PAUSE</button><button>REVIEW</button><button>MORE</button></section>
  </main>
 </div>
}







