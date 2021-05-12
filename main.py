# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:50:09 2021

@author: quantumliu
"""


import cv2
from cv2 import TrackerMOSSE_create
import numpy as np
from tqdm import tqdm
import subprocess
import os
import traceback

def pad_blank(img,x,y,w,h,color=(128,128,128)):
    img[y:y+h,x:x+w]=color

def main(path_video,path_out='out.mp4',scale=0.5):
# =============================================================================
#     path_video="F:\\Videos\\650nk 20210511\\merge.mp4"
# =============================================================================
    print(cv2.__version__)
    path_video=os.path.abspath(path_video)
    cap=cv2.VideoCapture(path_video)
    fps=cap.get(cv2.CAP_PROP_FPS)
    nb_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total:',str(nb_frame),str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer=cv2.VideoWriter('temp.mp4',cv2.VideoWriter_fourcc(*'MP4V'),fps,(w,h))
    
    
    
    tracker=TrackerMOSSE_create()
    flag,frame=cap.read()
    frame_scale=cv2.resize(frame,None,fx=scale,fy=scale)
    
    bb = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
    cv2.destroyAllWindows()
    
    initBB=tuple([int(i) for i in np.asarray(bb)*scale])
    x,y,w,h=initBB
    template=frame_scale[y:y+h,x:x+w].copy()
    h_tmp,w_tmp=template.shape[:2]
    
    tracker.init(frame_scale, initBB)
    pad_blank(frame,*bb)
    writer.write(frame)
    
    failed=False
    try:
        for _ in tqdm(range(nb_frame-1)):
            flag,frame=cap.read()
            if not flag:
                continue
            frame_scale=cv2.resize(frame,None,fx=scale,fy=scale)
            (success, box) = tracker.update(frame_scale)
            if success:
                bb=tuple([int(i) for i in np.asarray(box)/scale])
                pad_blank(frame,*bb)
            else:
                result = cv2.matchTemplate(frame_scale,template,cv2.TM_SQDIFF_NORMED)
                min_val, _, min_loc, _ = cv2.minMaxLoc(result)
                print(str(min_val))
                if min_val<0.2:
                    tracker=TrackerMOSSE_create()
                    bb=(*min_loc,w_tmp,h_tmp)
                    tracker.init(frame_scale, bb)
                    pad_blank(frame,*bb)
                
            writer.write(frame)
# =============================================================================
#         cv2.imshow('Preview',frame)
#         cv2.waitKey(1)
# =============================================================================
    except:
        failed=True
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        writer.release()
        cap.release()
    if not failed:
        cmd='''ffmpeg -i "temp.mp4" -i "{}" -map 0:v -map 1:a -c copy -shortest "{}" -y'''.format(path_video,path_out)
        subprocess.run(cmd,shell=True)
        os.remove("temp.mp4")
    
if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument("-v", "--video", type=str,help="path to input video file")
    ap.add_argument("-o", "--out",default='out.mp4', type=str,help="path to output video file")
    args = ap.parse_args()
    main(args.video,args.out)
# =============================================================================
#     main("F:\\Videos\\650nk 20210511\\merge.mp4")
# =============================================================================
