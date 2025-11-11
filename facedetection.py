import cv2
import time
import mediapipe as mp

class faceDetector:
    def __init__ (self, mode=False, maxFaces=2, detectionCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.detectionCon = detectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=0 if self.maxFaces == 1 else 1,
            min_detection_confidence=self.detectionCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        
        if self.results.detections:
            for llm in self.results.detections:
                if draw:
                    self.mpDraw.draw_detection(img, llm)

        return img
    
    def findPosition(self, img, faceNo=0, draw=True):
        bboxList = []
        if self.results.detections:
            myFace = self.results.detections[faceNo]
            ih, iw, ic = img.shape
            bboxC = myFace.location_data.relative_bounding_box
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            bboxList.append(bbox)
            if draw:        
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (255, 0, 255), 2)
        return bboxList
    


    
def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = faceDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        img = detector.findFaces(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])  # Print the position of the tip of the thumb

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


