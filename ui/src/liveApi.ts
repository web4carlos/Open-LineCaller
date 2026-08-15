export type LiveVideo={
 court_id:string; available:boolean; filename:string;
 fps:number|null; width:number|null; height:number|null;
};
export type OfficiatingSummary={
 available:boolean; event_count:number; source:string|null;
};
export type LiveCourt={
 court_id:string;name:string;status:string;mode:string;score:string;last_call:string;
 confidence:number|null;nearest_line:string;distance_in:number|null;evidence:string;
 serving_team:string;server_number:number|null;service_court:string;camera_id:string;session_id:string;
 video?:LiveVideo; officiating?:OfficiatingSummary;
};
export type LiveFacility={facility_name:string;sequence:number;courts:LiveCourt[]};

export type OfficiatingEvent={
 frame:number;
 final_decision:"IN"|"OUT"|"REVIEW";
 decision_confidence:number;
 image_x:number;
 image_y:number;
 nearest_line:string;
 signed_distance_in:number;
 geometry_state:string;
 bounce_confidence:number;
 bounce_score:number;
 gate_decision:string;
 gate_confidence:number;
 force_review:boolean;
 officiating_reason:string;
};

export type OfficiatingTimeline={
 court_id:string;
 count:number;
 events:OfficiatingEvent[];
 hold_frames:number;
 fps:number;
 video_width:number|null;
 video_height:number|null;
 source:string;
};

export const LIVE_BASE="http://127.0.0.1:8765";

export async function fetchLive():Promise<LiveFacility>{
 const r=await fetch(`${LIVE_BASE}/api/live`,{cache:"no-store"});
 if(!r.ok) throw new Error(`Live bridge HTTP ${r.status}`);
 return r.json();
}

export async function fetchOfficiating(courtId:string):Promise<OfficiatingTimeline>{
 const r=await fetch(`${LIVE_BASE}/api/officiating/${encodeURIComponent(courtId)}`,{cache:"no-store"});
 if(!r.ok) throw new Error(`Officiating HTTP ${r.status}`);
 return r.json();
}

export function courtVideoUrl(courtId:string){
 return `${LIVE_BASE}/api/video/${encodeURIComponent(courtId)}`;
}
