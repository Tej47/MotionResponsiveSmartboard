import mediapipe as mp
import cv2
import time

class finger_movement():
    def __init__(self, mode=False, maxHands=1, detectionConf=1, trackConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        
        
    def hand_recognition(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    
    
    def finger_tip(self, img, handNo=0, draw=True):
        self.listD = []
        if self.results.multi_hand_landmarks :
            myHand = self.results.multi_hand_landmarks[handNo]
            for id,lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h)
                self.listD.append([id,cx,cy])
                
                if draw:cv2.circle(img,(cx,cy),10,(255,0,255),-1)
        
        return self.listD

    def all_fingers(self):
        fingers=[]
        if self.listD[self.tipIds[0]-1][1]>self.listD[self.tipIds[0]][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        for id in range(1,5):
            if self.listD[self.tipIds[id]-2][2]>self.listD[self.tipIds[id]][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
