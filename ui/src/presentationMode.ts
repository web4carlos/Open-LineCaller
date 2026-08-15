export type PresentationMode="PLAYER"|"REFEREE";
export const DEFAULT_PRESENTATION_MODE:PresentationMode="PLAYER";
export function shouldShowDecision(mode:PresentationMode,d?:string|null){
 const x=String(d??"").toUpperCase();
 return mode==="REFEREE"?["IN","OUT","REVIEW"].includes(x):x==="OUT";
}
export function shouldSpeakDecision(_mode:PresentationMode,d?:string|null){
 return String(d??"").toUpperCase()==="OUT";
}
