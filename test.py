import cv2
import mediapipe as mp
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

# Colors.
blue = (255, 127, 0)
red = (245,66,230)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (245,117,66)
pink = (255, 0, 255)
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(0)
counter = 0 
stage = None
# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        # Detect stuff and render
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Make detection
        results = pose.process(image)
        
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract Landmarks
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

            l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]  
            l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]   

            r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]  
            r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]   

            # Calculate angle
            l_angle = round(calculate_angle(l_hip,l_knee,l_ankle),1)
            r_angle = round(calculate_angle(r_hip,r_knee,r_ankle),1)

            cv2.putText(image, str(l_angle),
                        tuple(np.multiply(l_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA
                            )
            
            cv2.putText(image, str(r_angle),
                        tuple(np.multiply(r_knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA
                            )

            if l_angle > r_angle+10 or r_angle > l_angle+10:
                cv2.putText(image, '!!!!!! No Balance !!!!!!',(150,50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
                
            # Draw landmarks.     
            cv2.circle(image, (lshoulder[0],lshoulder[1]), 7, yellow, -1)
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

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()