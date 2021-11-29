import cv2
import mediapipe as mp
import time

class DePose():
    def __init__(self):
        self.mpPose = mp.solutions.pose
        self.PoseDetect = self.mpPose.Pose()
        self.mpDraw = mp.solutions.drawing_utils

    def Detect(self,img, Draw=True):
        #img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.PoseDetect.process(imgRGB)

        if self.results.pose_landmarks:
            if Draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def FindPos(self,img,Draw = True):
        LdList = []
        if self.results.pose_landmarks:

            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx = int(w*lm.x)
                cy = int(h*lm.y)
                if Draw:
                    cv2.circle(img,(cx,cy),10,[255,255,0],cv2.FILLED)

                LdList.append([id,cx,cy])
        return img, LdList

def test():
    cap = cv2.VideoCapture(0)
    PD = DePose()
    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)
        img = PD.Detect(img)
        img, LdList = PD.FindPos(img, False)
        if len(LdList) > 0:
            cv2.circle(img, (LdList[3][1],LdList[3][2]), 10, [255, 255, 0], cv2.FILLED)
        #print(LdList[3])
        cv2.imshow('Image', img)
        if cv2.waitKeyEx(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test()


