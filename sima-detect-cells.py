from analysis import get_vid, df_over_f
import cv2
import numpy as np
from collections import defaultdict
from functools import partial
import sima
import matplotlib.pyplot as plt
import math
#import pyHook
#import pythoncom
#import runstats
#import scipy.misc


class ROISelector(object):
    def __init__(self, img,rois):
    	self.img = img.copy()        
    	self.coord = rois.copy()
    	self.wname = "roi"
    	self.done = False
    	self.newcoord = []
        
    def select_roi(self):
        cv2.imshow(self.wname, self.img)
        for (x,y) in self.coord:
        	cv2.circle(self.img, (x,y),3,(0,0,255),-1)
        cv2.setMouseCallback(self.wname, self.mouse_click_cb)
        while not self.done:
            cv2.imshow(self.wname, self.img)
            cv2.waitKey(33)
        #roi = self.roi.astype("uint8")
        #cv2.fillPoly(roi,np.array([self.verts]),1)        
        return self.newcoord

    def mouse_click_cb(self, event, x, y, flags, param):
        #lc = (255, 0, 0)
        #lw = 2
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.img, (x,y), 3, (0,0,255), -1)            
            self.newcoord.append((x,y))
        if event == cv2.EVENT_RBUTTONDOWN:            
            self.done = True

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(\
            description = "manually selects the frames that have cells",
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input", help="video")
    parser.add_argument("-r", "--raster", help="show raster", type = int)
    parser.add_argument("-o", "--output", help="output filename")
    parser.add_argument("-m", "--max-frame", help = "index of last frame", type=int)
    parser.add_argument('-s", "--start-frame', help = 'index of first frame', type=int)
    parser.add_argument('-ss', "--sima", help= "data coming from sima", type=int)

    args = parser.parse_args()
    if args.sima ==1:
    	dataset = sima.ImagingDataset.load(args.input[0])
    	print('extracting data from sima file')
    	for sequence in dataset:
    		seq = sequence
    	print('reshaping data')
    	vid = np.ravel(seq).reshape(seq.shape[0], seq.shape[2],seq.shape[3])
    else:
    	if args.max_frame and args.start_frame:
    		frame_shape,vid = get_vid([args.input], [args.max_frame], [args.start_frame])
    	else:
    		frame_shape, vid = get_vid([args.input])
    	vid = vid.T.reshape([vid.shape[-1]]+list(frame_shape))
    print(vid.shape)
    input('ready to play?')

    #frame_no = [21, 42, 55, 83, 96, 135, 171, 181, 232, 289, 313, 326, 391, 401, 416, 436]
    rois =[] # list of unique ROIS
    #time_rois = defaultdict(list)
    #cv2.namedWindow('video')    

    #cv2.setMouseCallback()
    #for i in range(500):
    #	cv2.imshow('video', vid[i])
    #	key = cv2.waitKey(100)
   # 	if key==ord('s'):
    #		frame_no =frame_no+[i]
    spikes = np.zeros((vid.shape[0],20))
    for i,frame in enumerate(vid):
    	cv2.imshow('video', frame)
    	key = cv2.waitKey(60)
    	if key==ord('s'):
    		roi_selector = ROISelector((frame*255).astype('uint8'),rois)    		
    		temp = roi_selector.select_roi()
    		for x,y in temp:
    			found = 0
    			for j,coord in enumerate(rois):
    				if math.sqrt(pow(coord[0]-x,2)+pow(coord[1]-y,2)) <=6:
    					spikes[i,j] = 1
    					found = 1
    			if found ==0:
    				spikes[i,len(rois)] = 1
    				rois = rois+ [(x,y)]
    				

    input('finished video, save rois?')
    np.savetxt("unique_rois-2017.txt", rois, delimiter = ' ', fmt='%i')
    with open(args.output, 'wb') as outf:
    	np.savez_compressed(outf, spikes)

    time = [i/20 for i in range(vid.shape[0])]
    if args.raster == 1:
    	for i in range(len(rois)):
    		plt.scatter(time, np.multiply(spikes[:,i],i+1))
    	plt.xlabel('Time (s)')
    	plt.ylabel('cell')
    	plt.show()



    #trace = np.zeros((vid.shape[0], len(rois)))
    #for i, frame in enumerate(vid):
    #	for j in range(len(rois)):
    #		trace[i,j] = np.mean(frame[rois[j][1]-3:rois[j][1]+3, rois[j][0]-3:rois[j][0]+3])    		


#    for i, frame in enumerate(vid):
#    	cv2.imshow('video', frame)
#    	cv2.waitKey(100)
#    	hm = pyHook.HookManager()
#    	hm.SubscribeMouseAllButtonsDown(onclick)
#    	hm.HookMouse()
#    	pythoncom.