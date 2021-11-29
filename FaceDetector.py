import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import os

class FaceDetector(object):
    def __init__(self,mtcnn,classifier):
        self.mtcnn=mtcnn
        self.classifier=classifier

    def _draw(self,frame,boxes,probs,landmarks,draw=True):
        try:
            for box, prob, ld in zip(boxes,probs,landmarks):
                cv2.rectangle(frame,
                              (box[0],box[1]),
                              (box[2],box[3]),
                              (0,0,255),
                              thickness=2)
            cv2.putText(frame, str(
                prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if draw==True:
                cv2.circle(frame, tuple(ld[0]), 6, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 6, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)

        except:
            pass

        return frame

    def _detect_ROIs(self,boxes):
        ROIs=list()
        for box in boxes:
            ROI=[int(box[1]), int(box[3]), int(box[0]), int(box[2])]
            ROIs.append(ROI)
        return ROIs

    def _blur_face(self,image,factor=3.0):
        (h,w) = image.shape[:2]
        KW=int(w/factor)
        KH=int(h/factor)

        if KW%2 == 0:
            KW -=1

        if KH%2 ==0:
            KH -=1

        return cv2.GaussianBlur(image,(KW,KH),0)

    def _is_it_shaobo(self,face):

        destRGB = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        PIL_img = Image.fromarray(destRGB.astype('uint8'), 'RGB')


        preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])

        processed_img = preprocess(PIL_img)
        batch_t = torch.unsqueeze(processed_img, 0)
        with torch.no_grad():
            out = self.classifier(batch_t)
            _, pred = torch.max(out, 1)

        prediction = np.array(pred[0])
        print(prediction)
        #shaobo=0 not_shaobo=1
        if prediction == 0:
            return('blur')
        else:
            return('dont_blur')



    def run(self, blur_setting=True):
        cap=cv2.VideoCapture(0) # 0 represents video from webcam  \ or you can put in the movie file name 'test.mp3'
        while True:
            ret,frame=cap.read() #read the next frame from the video, ret is whether the reading is successful or not
            try:
                boxes,probs,landmarks=self.mtcnn.detect(frame,landmarks=True)
                boxes=boxes.astype('int')
                landmarks=landmarks.astype('int')
                self._draw(frame,boxes,probs,landmarks)
                if blur_setting == True:
                    ROIs=self._detect_ROIs(boxes)
                    #print('test')
                    for roi in ROIs:
                        (startY,endY,startX,endX)=roi
                        face=frame[startY:endY,startX:endX]

                        pred=self._is_it_shaobo(face)
                        #print('test')
                        print(pred)

                        if pred == 'blur':
                            blured_face=self._blur_face(face)
                            frame[startY:endY, startX:endX]= blured_face
                        else:
                            pass

            except:
                pass

            cv2.imshow('Face Detection', frame)

            if cv2.waitKeyEx(1) & 0xFF==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()








