import threading

import cv2
from deepface import DeepFace

caption = cv2.VideoCapture("https://192.168.1.86:8080/video")

counter = 0
face_match_beto = False
face_match_beca = False

imagemBeto = cv2.imread("images/euTeste.png")
imagemBeca = cv2.imread("images/mae.jpg")

def check_face(frame):
    global face_match_beto, face_match_beca

    try:
        if DeepFace.verify(frame, imagemBeto.copy())['verified']:
            face_match_beto = True
        else:
            face_match_beto = False
    except ValueError:
        face_match_beto = False
    
    try:
        if DeepFace.verify(frame, imagemBeca.copy())['verified']:
            face_match_beca = True
        else:
            face_match_beca = False
    except ValueError:
        face_match_beca = False

while True:
    ret, frame = caption.read()

    if ret:
        if counter % 10 == 0:
            try:
                threading.Thread(target = check_face, args = (frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
    
        if face_match_beto:
            cv2.putText(frame, "ROBERTO", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        elif face_match_beca:
            cv2.putText(frame, "REBECA", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "DESCONHECIDO", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        cv2.imshow("Video", frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()