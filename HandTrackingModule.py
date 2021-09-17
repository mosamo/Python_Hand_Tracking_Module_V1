import cv2
import mediapipe as mp
import time


class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackingConfidence=0.5):
        # Hands Object parameters
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        # init and creation of hands object
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionConfidence, self.trackingConfidence)

        # init and creation of drawer
        self.mpDraw = mp.solutions.drawing_utils

    def draw_hands(self, img, draw=True):
        # media pipe hands requires rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        # if you found a hand
        if results.multi_hand_landmarks:
            # do for each hand:
            for handLandmarks in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLandmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def find_positions(self, img, handNumber=0, draw=True):

        landmark_List = []

        # media pipe hands requires rgb
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        # if you found a hand
        if results.multi_hand_landmarks:

            # SEE HandTracking.py TO SEE HOW TO EXECUTE FUNCTION FOR TWO HANDS, USE FOR LIST
            myHand = results.multi_hand_landmarks[handNumber]

            # get coordinate info for each hand vertex (i.e. landmark Node)
            # handLandmarks guide: https://miro.medium.com/max/875/1*JzJ_Ob4RfgsfEAbWtSbI6g.png
            for id, lm in enumerate(myHand.landmark):
                # each landmark vertex has x, y, z
                # values are normalized (0~1) so we need to multiply by width + height
                h, w, c = img.shape  # it is h,w not w,h
                cx, cy = int(lm.x * w), int(lm.y * h)

                landmark_List.append([id, cx, cy])
                # print coordinate info: print(id, ": ", cx, ",", cy)

        return landmark_List


def write_fps(img, pTime):
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    return cTime


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)

    detector = HandDetector()
    while True:
        success, img = cap.read()

        # using both draw_hands and find_positions is work intensive
        # 1. you should either combine them into one method
        #    o you don't find the vertex list twice
        # 2. only call one of them

        img = detector.draw_hands(img)
        landmarks_list = detector.find_positions(img)

        if len(landmarks_list) > 0:
            # print thumb coordinates
            print(landmarks_list[4])
            # draw circle on thumb
            idd, cx, cy = landmarks_list[4]
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        pTime = write_fps(img, pTime)
        cv2.imshow("Output", img)

        if cv2.waitKey(1) == ord('q'):
            # we wait (4) because video is loading too fast, if we process stuff then we can also delay it
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
