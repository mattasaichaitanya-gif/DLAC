"""Open first video frame and click polygon vertices; S to print polygon, R reset, Q quit."""
import cv2
from visionport.common import cli,settings,open_video,finish
args=cli('Click polygon points and press S to print JSON coordinates')
cfg=settings(args);cap=open_video(cfg);ok,frame=cap.read()
if not ok:raise RuntimeError('No frame available')
points=[]
def click(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:points.append([x,y])
cv2.namedWindow('Click polygon; S save coordinates; R reset; Q quit')
cv2.setMouseCallback('Click polygon; S save coordinates; R reset; Q quit',click)
try:
    while True:
        canvas=frame.copy()
        for p in points:cv2.circle(canvas,tuple(p),4,(0,255,255),-1)
        for a,b in zip(points,points[1:]):cv2.line(canvas,tuple(a),tuple(b),(0,255,255),2)
        cv2.imshow('Click polygon; S save coordinates; R reset; Q quit',canvas)
        key=cv2.waitKey(50)&0xff
        if key==ord('s'):print('Copy to settings.json:',points)
        if key==ord('r'):points=[]
        if key==ord('q') or key==27:break
finally:finish(cap)
