import cv2
from fer import FER


cap = cv2.VideoCapture(0)
# load the MTCCN model from FER
emo_detector = FER(mtcnn=True)

# set the font style
font= cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale= 1
fontColor= (255,255,255)
thickness= 1
lineType= 2

# set Line color and thickness
color = (255, 0, 0)
thickness = 2

while True:
    # start live camera capture mode
    success,img=cap.read()
    img=cv2.flip(img,1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # detect emotion using the FER module
    result = emo_detector.detect_emotions(imgRGB)
    dominant_emotion, emotion_score = emo_detector.top_emotion(imgRGB)

    # show the result
    bounding_box = result[0]["box"]
    emotions = result[0]["emotions"]
    cv2.rectangle(img, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  color, thickness )

    cv2.putText(img, ','.join([str(dominant_emotion), str(emotion_score)]),
                (bounding_box[0], bounding_box[1] + bounding_box[3] + 35),
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)

    cv2.imshow("Image", img)

    if cv2.waitKeyEx(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()