"""Local, non-generative passport extraction service.
BiRefNet weights are obtained once by rembg into U2NET_HOME and thereafter run locally by ONNX Runtime.
"""
import os, cv2, numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageFilter
from rembg import new_session, remove

app=FastAPI(title='local-passport-processor')
sessions={}
class Request(BaseModel):
 input_path:str; output_path:str; settings:dict
class Review(Exception): pass
def session():
 model=os.getenv('SEGMENTATION_MODEL','birefnet-general')
 if model not in sessions: sessions[model]=new_session(model)
 return sessions[model]
def quad_crop(bgr):
 """Find a plausible printed-photo quadrilateral, rectify perspective; leave photo alone if none is reliable."""
 h,w=bgr.shape[:2]; scale=min(1,1600/max(w,h)); small=cv2.resize(bgr,(int(w*scale),int(h*scale))) if scale<1 else bgr
 gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),45,140)
 cnts,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
 candidates=[]
 for c in cnts:
  area=cv2.contourArea(c); peri=cv2.arcLength(c,True); p=cv2.approxPolyDP(c,.02*peri,True)
  if len(p)==4 and area>.035*small.shape[0]*small.shape[1] and area<.88*small.shape[0]*small.shape[1]:
   x,y,rw,rh=cv2.boundingRect(p); ratio=rw/max(rh,1)
   if .45<ratio<1.8: candidates.append((area,p.reshape(4,2).astype('float32')/scale))
 if not candidates:return bgr,False
 pts=max(candidates,key=lambda x:x[0])[1]; s=pts.sum(1); d=np.diff(pts,axis=1).ravel(); ordered=np.array([pts[np.argmin(s)],pts[np.argmin(d)],pts[np.argmax(s)],pts[np.argmax(d)]],dtype='float32')
 tl,tr,br,bl=ordered; ow=int(max(np.linalg.norm(br-bl),np.linalg.norm(tr-tl))); oh=int(max(np.linalg.norm(tr-br),np.linalg.norm(tl-bl)))
 if min(ow,oh)<150:return bgr,False
 return cv2.warpPerspective(bgr,cv2.getPerspectiveTransform(ordered,np.float32([[0,0],[ow-1,0],[ow-1,oh-1],[0,oh-1]])),(ow,oh)),True
def enhance(rgb,enabled):
 im=Image.fromarray(rgb)
 if enabled:
  im=ImageEnhance.Color(im).enhance(1.03); im=ImageEnhance.Contrast(im).enhance(1.08); im=ImageEnhance.Sharpness(im).enhance(1.18)
  im=im.filter(ImageFilter.UnsharpMask(radius=1.2,percent=90,threshold=3))
 return np.asarray(im)
def validate(mask, out, target):
 coverage=float((mask>18).mean()); h,w=out.shape[:2]
 if coverage<.035 or coverage>.83: raise Review('Segmentation confidence is insufficient; use manual crop correction')
 if (w,h)!=(target[0],target[1]): raise Review('Output dimensions validation failed')
 # border should be nearly background white: an objective cleanliness check
 border=np.concatenate([out[:3].reshape(-1,3),out[-3:].reshape(-1,3),out[:,:3].reshape(-1,3),out[:,-3:].reshape(-1,3)])
 if border.mean()<185: raise Review('Background cleanliness validation failed')
@app.get('/health')
def health(): return {'ok':True,'model':'birefnet-general','cached':bool(sessions)}
@app.post('/process')
def process(req:Request):
 try:
  src=cv2.imread(req.input_path,cv2.IMREAD_COLOR)
  if src is None: raise HTTPException(422,'Unreadable image')
  s=req.settings; photo,detected=quad_crop(src) if s.get('autoDetect',True) else (src,False)
  rgb=cv2.cvtColor(photo,cv2.COLOR_BGR2RGB); rgb=enhance(rgb,s.get('enhance',True))
  # BiRefNet is a local ONNX inference session after its first cached download.
  rgba=np.asarray(remove(Image.fromarray(rgb),session=session()))
  if rgba.ndim!=3 or rgba.shape[2]!=4: raise Review('Person segmentation returned no alpha channel')
  alpha=rgba[:,:,3]; bg=s.get('background','#ffffff').lstrip('#'); color=tuple(int(bg[i:i+2],16) for i in (0,2,4)) if len(bg)==6 else (255,255,255)
  foreground=rgba[:,:,:3]; canvas=np.empty_like(foreground);canvas[:]=color
  a=(alpha.astype(np.float32)/255)[:,:,None]; composited=(foreground*a+canvas*(1-a)).astype(np.uint8)
  tw,th=int(s.get('width',600)),int(s.get('height',600)); scale=max(tw/composited.shape[1],th/composited.shape[0]); resized=cv2.resize(composited,(round(composited.shape[1]*scale),round(composited.shape[0]*scale)),interpolation=cv2.INTER_LANCZOS4)
  mask=cv2.resize(alpha,(resized.shape[1],resized.shape[0])); y=max(0,(resized.shape[0]-th)//3); x=max(0,(resized.shape[1]-tw)//2); out=resized[y:y+th,x:x+tw]; mask=mask[y:y+th,x:x+tw]
  if out.shape[:2]!=(th,tw): out=cv2.copyMakeBorder(out,0,max(0,th-out.shape[0]),0,max(0,tw-out.shape[1]),cv2.BORDER_CONSTANT,value=color); out=out[:th,:tw]
  validate(mask,out,(tw,th)); Path(req.output_path).parent.mkdir(parents=True,exist_ok=True)
  Image.fromarray(out).save(req.output_path,format='PNG' if s.get('format')=='png' else 'JPEG',quality=int(s.get('quality',90)),dpi=(int(s.get('dpi',300)),int(s.get('dpi',300))))
  return {'ok':True,'detectedPhoto':detected,'manualReview':False,'width':tw,'height':th}
 except Review as e: return {'ok':False,'manualReview':True,'error':str(e)}
 except HTTPException: raise
 except Exception as e: return {'ok':False,'manualReview':True,'error':f'Processing error: {str(e)[:250]}'}
