from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import numpy as np
import cv2
import base64
from django.views.decorators.csrf import csrf_exempt
import dlib
from scipy.spatial import distance as dist
import os

def home(request):
    return render(request,'index.html');

def login(request):
    return render(request,'login.html')

def login_check(request):
    username = request.GET["username"]
    password = request.GET["password"]

    if username=="admin" and password=="admin":
        return render(request,'login.html',{'log':'Login successful'})
    else:
        return render(request, 'login.html', {'log': 'Incorrect credentials'})

def eye_aspect_ratio(eye):
    # compute the euclidean distance between the vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal eye landmarks
    C = dist.euclidean(eye[0], eye[3])

    # compute the EAR
    ear = (A + B) / (2 * C)
    return ear


class wink:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(os.path.join(os.getcwd(),'blinkapp\shape.dat'))
        self.JAWLINE_POINTS = list(range(0, 17))
        self.RIGHT_EYEBROW_POINTS = list(range(17, 22))
        self.LEFT_EYEBROW_POINTS = list(range(22, 27))
        self.NOSE_POINTS = list(range(27, 36))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.MOUTH_OUTLINE_POINTS = list(range(48, 61))
        self.MOUTH_INNER_POINTS = list(range(61, 68))

        self.EYE_AR_THRESH = 0.22
        self.EYE_AR_CONSEC_FRAMES = 3
        self.EAR_AVG = 0
        self.COUNTER = 0
        self.TOTAL = 0

    def detect_wink(self,frame):




        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        for rect in rects:
            x = rect.left()
            y = rect.top()
            x1 = rect.right()
            y1 = rect.bottom()
            # get the facial landmarks
            landmarks = np.matrix([[p.x, p.y] for p in self.predictor(frame, rect).parts()])
            # get the left eye landmarks
            left_eye = landmarks[self.LEFT_EYE_POINTS]
            # get the right eye landmarks
            right_eye = landmarks[self.RIGHT_EYE_POINTS]
            # draw contours on the eyes
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0),
                             1)  # (image, [contour], all_contours, color, thickness)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            # compute the EAR for the left eye
            ear_left = eye_aspect_ratio(left_eye)
            # compute the EAR for the right eye
            ear_right = eye_aspect_ratio(right_eye)
            # compute the average EAR
            ear_avg = (ear_left + ear_right) / 2.0
            # detect the eye blink
            if ear_avg < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1

                    print("Eye blinked")
                self.COUNTER = 0

            cv2.putText(frame, "Blinks{}".format(self.TOTAL), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
            cv2.putText(frame, "EAR {}".format(ear_avg), (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 1)
        return frame
print(os.path.realpath(__file__))
imgs = wink()
@csrf_exempt
def img_from_web(request):
    image_data = request.POST["img"]
    base64_img_bytes = image_data.encode('utf-8')
    decoded_image_data = base64.decodebytes(base64_img_bytes)
    npimg = np.fromstring(decoded_image_data, dtype=np.uint8)
    img=cv2.imdecode(npimg, 1)
    img= np.array(img,dtype=np.uint8)

    pimg = imgs.detect_wink(img)

    retval, buffer = cv2.imencode('.png', pimg)
    jpg_as_text = base64.b64encode(buffer)
    jpg_as_text = jpg_as_text.decode("utf-8")

    return JsonResponse({'image':str(jpg_as_text)})