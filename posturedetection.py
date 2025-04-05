import cv2
import numpy as np
import mediapipe as mp
import serial.tools.list_ports

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

bad_posture_frames = 0
ALERT_THRESHOLD = 50

font = cv2.FONT_HERSHEY_SIMPLEX

blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def find_angles(vec1, vec2, vec3):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    vec3 = np.array(vec3)

    vec21 = vec1 - vec2
    vec23 = vec3 - vec2

    dist_vec21 = distance(vec1, vec2)
    dist_vec23 = distance(vec3, vec2)

    if dist_vec21 == 0 or dist_vec23 == 0:
        return 0

    angle = np.degrees(np.arccos((np.dot(vec21, vec23)) / (dist_vec21 * dist_vec23)))

    return angle

liveFeed = cv2.VideoCapture(0)

ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()
portsList = []

for one in ports:
    portsList.append(str(one))
    print(str(one))

com = input("Select Com Port for Arduino #: ")

for i in range(len(portsList)):
    if portsList[i].startswith("COM" + str(com)):
        print("COM" + str(com))

serialInst.baudrate = 9600
serialInst.port = "COM" + str(com)
serialInst.open()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while liveFeed.isOpened():
        _, img = liveFeed.read()

        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgbImage)
        rs = results.pose_landmarks

        if rs:
            landmarks = rs.landmark

            leftKnee = [landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_holistic.PoseLandmark.LEFT_KNEE.value].y]

            leftHip = [landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y]

            leftShoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]

            leftEar = [landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].x,
                       landmarks[mp_holistic.PoseLandmark.LEFT_EAR.value].y]

            if all(landmark is not None for landmark in [leftKnee, leftHip, leftShoulder, leftEar]):
                virtualPoint = [leftEar[0], leftEar[1] - 0.1]  # Normalized coords
                angleKHS = find_angles(leftKnee, leftHip, leftShoulder)
                angleHSE = find_angles(leftHip, leftShoulder, leftEar)
                angleSEV = find_angles(leftShoulder, leftEar, virtualPoint)

                validKHS = angleKHS >= 75 and angleKHS <= 105
                validHSE = angleHSE >= 165
                validSEV = angleSEV >= 165

                if validKHS and validHSE and validSEV:
                    bad_posture_frames = 0
                    cv2.putText(img, "GOOD POSTURE!", (350, 50), font, 1, green, 2)
                else:
                    bad_posture_frames += 1

                    # Specific feedback based on which angle is wrong
                    if not validKHS:
                        cv2.putText(img, "SIT UP STRAIGHT!", (350, 50), font, 1, red, 2)
                    elif not validHSE:
                        cv2.putText(img, "FIX SHOULDER POSITION!", (350, 50), font, 1, red, 2)
                    else:  # Only SEV is invalid
                        cv2.putText(img, "FIX NECK POSITION!", (350, 50), font, 1, red, 2)

                    # General alert if bad posture persists
                    if bad_posture_frames > ALERT_THRESHOLD:
                        cv2.putText(img, "BAD POSTURE ALERT!", (50, 350), font, 1.5, red, 3)
                        command = "BP"
                        serialInst.write(command.encode('utf-8'))


                # Draw angles on screen
                cv2.putText(img, f"KHS: {angleKHS:.1f}°", (10, 30), font, 0.7, green, 2)
                cv2.putText(img, f"HSE: {angleHSE:.1f}°", (10, 60), font, 0.7, blue, 2)
                cv2.putText(img, f"SEV: {angleSEV:.1f}°", (10, 90), font, 0.7, pink, 2)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=green, thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=yellow, thickness=2)
            )

        cv2.imshow('Posture Check', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

liveFeed.release()
cv2.destroyAllWindows()
