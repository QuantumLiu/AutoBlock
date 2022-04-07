# -*- coding: utf-8 -*-
"""
Created on Tue May 11 21:50:09 2021

@author: quantumliu
"""


import cv2
from cv2 import TrackerKCF_create
import numpy as np
from tqdm import tqdm
import subprocess
import os
import traceback

def pad_blank(img,x,y,w,h,color=(128,128,128)):
    img[y:y+h,x:x+w]=color

def rotate_bound(image, angle):#图片旋转但不改变大小，模板匹配中大小改变对匹配效果有影响
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)#//是向下取整
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH))


def main(path_video,display,name_out='out.mp4',scale=0.25):
    print(cv2.__version__)
    path_video=os.path.abspath(path_video)
    path_dir=os.path.dirname(path_video)
    path_temp=os.path.join(path_dir,'temp.mp4')
    path_out=os.path.join(path_dir,name_out)
    cap=cv2.VideoCapture(path_video)
    fps=cap.get(cv2.CAP_PROP_FPS)
    nb_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total:',str(nb_frame),str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    writer=cv2.VideoWriter(path_temp,cv2.VideoWriter_fourcc(*'avc1'),fps,(w,h))
    
    preivew=display
    
    
    tracker=TrackerKCF_create()
    flag,frame=cap.read()
    frame_scale=cv2.resize(frame,None,fx=scale,fy=scale)
    
    cv2.namedWindow("Frame",cv2.WINDOW_AUTOSIZE)
    bb = cv2.selectROI("Frame", frame_scale, fromCenter=False,showCrosshair=True)
    bb=tuple([int(i) for i in np.asarray(bb)/scale])
    cv2.destroyAllWindows()
    
    initBB=tuple([int(i) for i in np.asarray(bb)*scale])
    x,y,w,h=initBB

    x,y,w,h=bb
    area_origin=w*h
    print(bb)
    template_sift=cv2.convertScaleAbs(frame,alpha=1.0)[y:y+h,x:x+w]

    MIN_MATCH_COUNT = 6#设置最低特征点匹配数量为10
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 20)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    template_gray=cv2.cvtColor(template_sift,cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template_gray,None)

    
    tracker.init(frame_scale, initBB)
    pad_blank(frame,*bb)
    writer.write(frame)
    
    failed=False
    n_lose=0
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
                if display and preivew:
                    pad_blank(frame_scale,*box)
                n_lose=0
            else:
                if n_lose>10:
                    cv2.destroyAllWindows()
                    print("Refind failed, please select ROI again.")
                    cv2.namedWindow("Select ROI",cv2.WINDOW_AUTOSIZE)
                    bb = cv2.selectROI("Select ROI", frame_scale, fromCenter=False,showCrosshair=True)
                    bb=tuple([int(i) for i in np.asarray(bb)/scale])
                    cv2.destroyAllWindows()
                    preivew=True
                    
                    initBB=tuple([int(i) for i in np.asarray(bb)*scale])
                    tracker=TrackerKCF_create()
                    tracker.init(frame_scale, initBB)
                    pad_blank(frame,*bb)
                    writer.write(frame)
                    continue

                preivew=True
                n_lose+=1
                frame_trans=cv2.convertScaleAbs(frame,alpha=1.0)
                target=cv2.cvtColor(frame_trans,cv2.COLOR_BGR2GRAY)
                kp2, des2 = sift.detectAndCompute(target,None)

                matches = flann.knnMatch(des1,des2,k=2)
                matchesMask=[[0,0] for i in range (len(matches))]
                for i, (m,n) in enumerate(matches):
                    if m.distance< 0.6*n.distance: #舍弃小于0.7的匹配结果
                        matchesMask[i]=[1,0]

                if display:
                    drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0) #给特征点和匹配的线定义颜色
                    resultimage=cv2.drawMatchesKnn(template_gray,kp1,frame_trans,kp2,matches,None,**drawParams) #画出匹配的结果
                    cv2.namedWindow("resultimage",cv2.WINDOW_AUTOSIZE)
                    cv2.imshow("resultimage",resultimage)

                good = []
                #舍弃大于0.7的匹配
                for m,n in matches:
                    if m.distance < 0.6*n.distance:
                        good.append(m)
                print('Target lossed.\nDoing FLANN match to detect it again......\n')
                print("Found {}/{} matched points".format(len(good),MIN_MATCH_COUNT))
                if len(good)>=MIN_MATCH_COUNT:
                    # 获取关键点的坐标
                    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                    #计算变换矩阵和MASK
                    M, mask = cv2.findHomography(src_pts, dst_pts)
                    matchesMask = mask.ravel().tolist()
                    h,w = template_gray.shape
                    # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
                    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                    try:
                        dst = cv2.perspectiveTransform(pts,M)
                        if display:
                            cv2.polylines(frame_scale,[np.int32(dst*scale)],True,0,2, cv2.LINE_AA)
                            cv2.namedWindow("frame_scale",cv2.WINDOW_AUTOSIZE)
                            cv2.imshow('frame_scale',frame_scale)

                    except :
                        print(pts)
                        print(M)
                        cv2.waitKey()
                    x,y,w,h = cv2.boundingRect(dst)
                    if w*h<4*area_origin and w*h<area_origin/4:
                        bb=(x,y,w,h)
                        print("Re find succesed!")
                        cv2.destroyAllWindows()

                        pad_blank(frame,*bb)
                        bb=tuple([int(i) for i in np.asarray(bb)*scale])
                        if display and preivew:
                            pad_blank(frame_scale,*box)
                        tracker=TrackerKCF_create()
                        tracker.init(frame_scale, bb)
                    
            if display and preivew:
                cv2.namedWindow("preview",cv2.WINDOW_AUTOSIZE)
                cv2.imshow('preview',frame_scale)
                k=cv2.waitKey(1)
                if k==27:
                    preivew=False
                    cv2.destroyWindow('preview')
            writer.write(frame)

    except:
        failed=True
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        writer.release()
        cap.release()
    if not failed:
        cmd='''ffmpeg -i "{}" -i "{}" -map 0:v -map 1:a -c copy -shortest "{}" -y'''.format(path_temp,path_video,path_out)
        subprocess.run(cmd,shell=True)
        os.remove(path_temp)
    
if __name__=='__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Process some integers.')
    ap.add_argument("-v", "--video", type=str,help="path to input video file")
    # ap.add_argument("-o", "--out",default='out.mp4', type=str,help="path to output video file")
    ap.add_argument("-d", "--display",action='store_true',default=False)
    ap.add_argument("-o","--name_out",type=str,default="out.mp4")
    ap.add_argument("--scale",type=float,default=0.5)
    args = ap.parse_args()
    main(args.video,args.display,args.name_out,args.scale)
