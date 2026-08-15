import {useEffect,useMemo,useRef,useState} from "react";
import {
  fetchOfficiating,
  LiveCourt,
  OfficiatingEvent,
  OfficiatingTimeline,
  courtVideoUrl,
} from "./liveApi";

type Props={
 court:LiveCourt;
 onEventChange?:(event:OfficiatingEvent|null,frame:number)=>void;
};

function eventAt(t:OfficiatingTimeline|null,frame:number){
 if(!t) return null;
 let selected:OfficiatingEvent|null=null;
 for(const e of t.events){
   if(e.frame>frame) break;
   selected=e;
 }
 if(!selected) return null;
 return frame<=selected.frame+t.hold_frames?selected:null;
}

export default function LiveCourtVideo({court,onEventChange}:Props){
 const[failed,setFailed]=useState(false);
 const[timeline,setTimeline]=useState<OfficiatingTimeline|null>(null);
 const[active,setActive]=useState<OfficiatingEvent|null>(null);
 const[frame,setFrame]=useState(0);
 const videoRef=useRef<HTMLVideoElement|null>(null);
 const available=Boolean(court.video?.available)&&!failed;

 useEffect(()=>{
   let alive=true;

   if(!court.officiating?.available){
     setTimeline(null);
     return;
   }

   fetchOfficiating(court.court_id)
     .then(x=>{
       if(!alive) return;

       setTimeline(x);

       const video=videoRef.current;

       if(video){
         video.pause();
         video.currentTime=0;

         setTimeout(()=>{
           video.currentTime=0;
           video.play().catch(()=>{});
         },150);
       }
     })
     .catch(()=>{
       if(alive)setTimeline(null);
     });

   return()=>{alive=false};
 },[court.court_id,court.officiating?.available]);

 function sync(){
   const v=videoRef.current;
   if(!v) return;
   const fps=timeline?.fps||court.video?.fps||60;
   const current=Math.floor(v.currentTime*fps);
   const next=eventAt(timeline,current);
   setFrame(current);
   setActive(next);
   onEventChange?.(next,current);
 }

 const marker=useMemo(()=>{
   if(!active) return null;
   const w=timeline?.video_width||court.video?.width;
   const h=timeline?.video_height||court.video?.height;
   if(!w||!h) return null;
   return {
     left:`${(active.image_x/w)*100}%`,
     top:`${(active.image_y/h)*100}%`
   };
 },[active,timeline,court.video]);

 return <div className="video">
   {available ? <>
     <video
       ref={videoRef}
       className="realCourtVideo"
       src={courtVideoUrl(court.court_id)}
       muted loop playsInline controls
       onTimeUpdate={sync}
       onSeeked={sync}
       onError={()=>setFailed(true)}
     />
     <div className="realVideoShade"/>

     {active && <div className={"syncedDecision "+active.final_decision.toLowerCase()}>
       <b>{active.final_decision}</b>
       <span>{Math.round(active.decision_confidence*100)}%</span>
     </div>}

     {marker && <div className="bounceMarker" style={marker}>
       <i/>
       <span>BOUNCE &middot; F{active?.frame}</span>
     </div>}
   </> : <div className="videoNoSource">
     <div className="cameraGlyph">&#9673;</div>
     <b>REAL VIDEO SOURCE NOT CONFIGURED</b>
     <span>Restart Python bridge with --video C:\path\match.mp4</span>
   </div>}

   <div className="videoChrome">
     <span>{court.camera_id||"NO CAMERA"} &middot; {available?"VIDEO LIVE":"WAITING VIDEO"}</span>
     <span>{court.video?.filename||"NO SOURCE"}</span>
   </div>

   <div className="videoBottom">
     <span>FRAME {frame}</span>
     <span>{timeline?`${timeline.count} FINAL OFFICIATING EVENTS`:"NO EVENT TIMELINE"}</span>
     <span>{court.mode}</span>
   </div>
 </div>
}






