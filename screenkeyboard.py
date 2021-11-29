import cv2
import mediapipe as mp
from pynput.keyboard import Controller
import time
import pyglet

cap = cv2.VideoCapture(0)
pyglet.options['search_local_libs'] = True
sound = pyglet.resource.media("click.wav", streaming=False)
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)

# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

Keys = [['Q','W','E','R','T','Y','U','I','O','P'],
       ['A','S','D','F','G','H','J','K','L',';'],
       ['Z','X','C','V','B','N','M',',','DD']]

KeyBoard = Controller()
#KeyBoard.press('a')
ButtonList=[]

def DrawAll(img,ButtonList,TipP8,TipP12,Hand=False):
    h, w, c = img.shape
    x8=0
    y8=0
    text = ''
    if Hand:
        x8 = w*TipP8.x
        y8 = h*TipP8.y

        x12 = w * TipP12.x
        y12 = h * TipP12.y


    for i, button in enumerate(ButtonList):
        x, y = button.pos
        w, h = button.size
        if x< x8 < x+w and y< y8<y+h:
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            if x< x12 < x+w and y< y12 <y+h:
                cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 0), cv2.FILLED)
                text=button.text

        else:
            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 0, 255), cv2.FILLED)

        cv2.putText(img, button.text, (button.pos[0] + 23, button.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255, 255, 255), 3)
    # if len(K) >0 :
    #     x, y = ButtonList[K[0]].pos
    #     w, h = ButtonList[K[0]].size
    #     if Click == True:
    #         cv2.rectangle(img, [x,y], (x + w, y + h), (0, 255, 255), cv2.FILLED)
    #     else:
    #         cv2.rectangle(img, [x, y], (x + w, y + h), (0, 0, 255), cv2.FILLED)
    #     cv2.putText(img, ButtonList[K[0]].text, (x + 23, y + 60), cv2.FONT_HERSHEY_PLAIN, 5,
    #                 (255, 255, 255), 3)
    return img,text

class Button():
    def __init__(self, pos, text, size=[90, 90]):
        self.pos = pos
        self.text = text
        self.size = size

for i in range(len(Keys)):
    for j,Key in enumerate(Keys[i]):
        ButtonList.append(Button([100*j+50, 100*i+50], Key))

textp=''
timep=0
finaltext=''

while True:

    success,img=cap.read()
    img=cv2.flip(img,1)
    h, w, c = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #face = face_mesh.process(imgRGB)
    #print(results.multi_hand_landmarks[0].landmarks[0])
    TipP8=[]
    TipP12=[]
    Hand=False
    timec=0

    if results.multi_hand_landmarks:
        TipP8 = results.multi_hand_landmarks[0].landmark[8]
        TipP12 = results.multi_hand_landmarks[0].landmark[12]
        cv2.circle(img, [int(w*TipP8.x),int(h*TipP8.y)], 5, [0, 255, 0], 21)
        Hand = True
        for lm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,lm, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color =(0,255,0),thickness=2,circle_radius=2),
                                  mpDraw.DrawingSpec(color =(100,155,0),thickness=2,circle_radius=2))


    #img = detector.findHands(img)
    #lmList,bboxInfo = detector.find(img)

    img, text =DrawAll(img,ButtonList,TipP8,TipP12,Hand)
    timec = time.time()
    if text == 'DD':
        finaltext = ''
    else:
        if textp != text:
            textp = text
            timep = timec
            if len(text) > 0:
                finaltext += text
                sound.play()
        else:
            if timec-timep > 1:
                timep = timec
                if len(text) > 0:
                    finaltext += text
                    sound.play()
    cv2.rectangle(img, [100,500],[600,600],[0,255,255])
    cv2.putText(img,finaltext,(120,580), cv2.FONT_HERSHEY_PLAIN, 5,
                        (255, 255, 255), 3)

    cv2.imshow("Image", img)

    if cv2.waitKeyEx(1) & 0xFF==ord('q') or text =='q':
        break

cap.release()
cv2.destroyAllWindows()