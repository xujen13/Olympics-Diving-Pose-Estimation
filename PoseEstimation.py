import mediapipe as mp
import cv2 as cv 
import numpy as np
import pandas as pd
import time

video_path = r"C:\Users\Karafuru\Desktop\Forward School\ADS\Capstone\test.mp4"

class PoseDetector:

    def __init__(self, mode = False, upBody = False, smooth=True, detectionCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = 1

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.modelComplex, self.smooth, self.detectionCon, self.trackCon)

        
    def findPose(self, im, draw=True):
        rgb = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        self.results = self.pose.process(rgb)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(im, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return im

    def findPosition(self, im, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = im.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv.circle(im, (cx, cy), 5, (255, 0, 0), cv.FILLED)
        return lmList

    
def main():
    cap = cv.VideoCapture(video_path) #make VideoCapture(0) for webcam
    pTime = 0
    detector = PoseDetector()
    data = []

    while True:
        success, im = cap.read()
        if success:
            im = detector.findPose(im)
            lmList = detector.findPosition(im)
            print(lmList)
            data.extend(lmList)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            cv.putText(im, str(int(fps)), (60, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 3)
            
            cv.imshow("Image", im)
            cv.waitKey(10)

            #lmList = pd.DataFrame(data = lmList)
            #lmList.to_csv("./dataset1.csv")

        else:
            break
        
    data_df = pd.DataFrame(columns=data)
    data_df.to_csv("./dataset1.csv")

if __name__ == "__main__":
    main()

