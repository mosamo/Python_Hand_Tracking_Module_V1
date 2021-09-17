import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# init and creation of hands object
mpHands = mp.solutions.hands
hands = mpHands.Hands()  # ctrl+click to see params, defaults are good

# init and creation of drawer
mpDraw = mp.solutions.drawing_utils

pTime = 0

def write_fps(img, pTime):
    cTime = time.time()
    fps = 1/(cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return cTime

while True:
    success, img = cap.read()

    # mediapipe hands needs rgb
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        # for each hand
        for handLandmarks in results.multi_hand_landmarks:
            # handLandmarks guide: https://miro.medium.com/max/875/1*JzJ_Ob4RfgsfEAbWtSbI6g.png
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)
            # get hand info for each vertice
            for id, lm in enumerate(handLandmarks.landmark):
                # each lm has x, y, z
                # values are normalized so we need to multiply by width + height
                h, w, c = img.shape # it is h,w not w,h
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, ": ", cx, ",", cy)
                ''' draw circle on thumb tip
                if id == 4:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)'''

    ''' FPS DISPLAY '''
    pTime = write_fps(img, pTime)
    '''             '''

    cv2.imshow("Output", img)

    if cv2.waitKey(1) == ord('q'):
        # we wait (4) because video is loading too fast, if we process stuff then we can also delay it
        cv2.destroyAllWindows()
        break