import tkinter as tk 
import customtkinter as ck 

import pandas as pd 
import numpy as np 
import pickle 

import mediapipe as mp
import cv2
from PIL import Image, ImageTk 
from landmarks import landmarks

blue = (255, 127, 0)
red = (245,66,230)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (245,117,66)
pink = (255, 0, 255)

window = tk.Tk()
window.geometry("480x700")
window.title("GYMMY")
ck.set_appearance_mode("dark")
window.configure(bg='white')

classLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
classLabel.place(x=10, y=1)
classLabel.configure(text='STAGE') 
counterLabel = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
counterLabel.place(x=160, y=1)
counterLabel.configure(text='REPS') 
probLabel  = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", padx=10)
probLabel.place(x=300, y=1)
probLabel.configure(text='PROB') 
classBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", fg_color="#99CCFF")
classBox.place(x=10, y=41)
classBox.configure(text='0') 
counterBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", fg_color="#99CCFF")
counterBox.place(x=160, y=41)
counterBox.configure(text='0') 
probBox = ck.CTkLabel(window, height=40, width=120, text_font=("Arial", 20), text_color="black", fg_color="#99CCFF")
probBox.place(x=300, y=41)
probBox.configure(text='0') 

def reset_counter(): 
    global counter
    counter = 0 

button = ck.CTkButton(window, text='RESET', command=reset_counter, height=40, width=120, text_font=("Arial", 20), text_color="white", fg_color="#99CCFF")
button.place(x=10, y=600)

frame = tk.Frame(height=480, width=480)
frame.place(x=10, y=90) 
lmain = tk.Label(frame) 
lmain.place(x=0, y=0) 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5) 

with open('coords4.pkl', 'rb') as f: 
    model = pickle.load(f) 

cap = cv2.VideoCapture(0)
current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 

def detect(): 
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob 

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,dsize = (480,480),fx=0.2,fy=0.2,interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = pose.process(image)
    try:
            landmarks = results.pose_landmarks.landmark

            lshoulder = [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h)]
            lhip = [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h)]  
            lknee = [int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h)]
            lankle = [int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h)]       
            lheel = [int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y * h)]         
            lfoot = [int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y * h)]
                       
            rshoulder = [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h)]       
            rhip = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h)]           
            rknee = [int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h)]
            rankle = [int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w),
                    int(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h)]       
            rheel = [int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y * h)]         
            rfoot = [int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x * w),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y * h)]  

            # Draw landmarks.     
            cv2.circle(image, (lshoulder[0],lshoulder[1]), 7, yellow, -1, cv2.LINE_AA)
            cv2.circle(image, (lhip[0],lhip[1]), 7, yellow, -1)
            cv2.circle(image, (lknee[0],lknee[1]), 7, yellow, -1)
            cv2.circle(image, (lankle[0],lankle[1]), 7, yellow, -1)
            cv2.circle(image, (lheel[0],lheel[1]), 7, yellow, -1)
            cv2.circle(image, (lfoot[0],lfoot[1]), 7, yellow, -1)

            cv2.circle(image, (rshoulder[0],rshoulder[1]), 7, yellow, -1)
            cv2.circle(image, (rhip[0],rhip[1]), 7, yellow, -1)
            cv2.circle(image, (rknee[0],rknee[1]), 7, yellow, -1)
            cv2.circle(image, (rankle[0],rankle[1]), 7, yellow, -1)
            cv2.circle(image, (rheel[0],rheel[1]), 7, yellow, -1)
            cv2.circle(image, (rfoot[0],rfoot[1]), 7, yellow, -1)
            
            # Join landmarks.
            cv2.line(image, (lshoulder[0],lshoulder[1]), (lhip[0],lhip[1]), red, 4)
            cv2.line(image, (lshoulder[0],lshoulder[1]), (rshoulder[0],rshoulder[1]), red, 4)
            cv2.line(image, (lhip[0],lhip[1]), (rhip[0],rhip[1]), red, 4)
            cv2.line(image, (lknee[0],lknee[1]), (lhip[0],lhip[1]), red, 4)
            cv2.line(image, (rshoulder[0],rshoulder[1]), (rhip[0],rhip[1]), red, 4)
            cv2.line(image, (rknee[0],rknee[1]), (rhip[0],rhip[1]), red, 4)
            
            cv2.line(image, (lknee[0],lknee[1]), (lankle[0],lankle[1]), red, 4)
            cv2.line(image, (lankle[0],lankle[1]), (lheel[0],lheel[1]), red, 4)
            cv2.line(image, (lfoot[0],lfoot[1]), (lheel[0],lheel[1]), red, 4)
            cv2.line(image, (lfoot[0],lfoot[1]), (lankle[0],lankle[1]), red, 4)

            cv2.line(image, (rknee[0],rknee[1]), (rankle[0],rankle[1]), red, 4)
            cv2.line(image, (rankle[0],rankle[1]), (rheel[0],rheel[1]), red, 4)
            cv2.line(image, (rfoot[0],rfoot[1]), (rheel[0],rheel[1]), red, 4)
            cv2.line(image, (rfoot[0],rfoot[1]), (rankle[0],rankle[1]), red, 4)
            
    except:
        pass

    try: 
        row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
        X = pd.DataFrame([row], columns = landmarks) 
        bodylang_prob = model.predict_proba(X)[0]
        bodylang_class = model.predict(X)[0] 

        if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.6: 
            current_stage = "down" 
        elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.6:
            current_stage = "up" 
            counter += 1 

    except Exception as e: 
        print(e) 

    img = image[:, :460, :] 
    imgarr = Image.fromarray(img) 
    imgtk = ImageTk.PhotoImage(imgarr) 
    lmain.imgtk = imgtk 
    lmain.configure(image=imgtk)
    lmain.after(10, detect)  

    counterBox.configure(text=counter) 
    probBox.configure(text=bodylang_prob[bodylang_prob.argmax()]) 
    classBox.configure(text=current_stage) 

detect() 
window.mainloop()