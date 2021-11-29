import cv2
import mediapipe as mp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import matplotlib.animation
from IPython.display import clear_output

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#separate window for abstracted landmarks
plt.ion()
fig = plt.figure()

#ax = plt.axes(projection='3d')
xdata=[]
ydata=[]
zdata=[]

ax = fig.add_subplot(111)
scatter, = ax.plot(xdata, ydata,'g.')

while True:
    success,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    face = face_mesh.process(imgRGB)
    #print(results.multi_hand_landmarks[0].landmarks[0])

    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,lm, mpHands.HAND_CONNECTIONS,
                                  mpDraw.DrawingSpec(color =(0,255,0),thickness=2,circle_radius=2),
                                  mpDraw.DrawingSpec(color =(100,155,0),thickness=2,circle_radius=2))

    if face.multi_face_landmarks:
        #print(face.multi_face_landmarks.landmarks[0])
        #print(face.multi_face_landmarks[0].landmark[0])
        xdata=[]
        ydata=[]
        zdata=[]

        for face_landmarks in face.multi_face_landmarks:
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mpDraw.draw_landmarks(
                image=img,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            #print(face_landmarks.landmark[1].x)
            for ld in face_landmarks.landmark:
                xdata.append(ld.x)
                ydata.append(1-ld.y)
                zdata.append(ld.z)


        x = face.multi_face_landmarks[0].landmark[4].x
        y = face.multi_face_landmarks[0].landmark[4].y

        h,w,c = img.shape

        x=int(w*x)
        y=int(h*y)

        #print(type(face.multi_face_landmarks[0].landmark))
        cv2.circle(img,[x,y],5,[0,255,0],21)

        ax.clear()
        ax.set_aspect(h/w)
        #ax.set_xlim(0, 1)
        #ax.set_ylim(0, 1)
        s = [1] * len(xdata)
        plt.scatter(xdata,ydata,s,c=zdata)
        plt.axis('off')
        plt.show()

    #img = detector.findHands(img)
    #lmList,bboxInfo = detector.find(img)

    cv2.imshow("Image", img)

    if cv2.waitKeyEx(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()