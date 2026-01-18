import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import time

wCam, hCam = 640, 480
frameR = 100
smoothening = 7
click_cooldown = 0.5
click_distance = 30
SCROLL_SENSITIVITY = 1.2
MOVEMENT_THRESHOLD = 8
DRAG_ACTIVATION_TIME = 0.3

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

prev_x, prev_y = pyautogui.position()
last_click_time = 0
scroll_active = False
lock_mouse_pos = None
dragging = False
drag_start_time = 0

def get_landmarks(img, results):
    landmarks = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w = img.shape[:2]
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))
            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
            )
    return landmarks

def check_fingers(landmarks):
    fingers = []
    if landmarks[0][1] < landmarks[17][1]:
        fingers.append(1 if landmarks[4][1] > landmarks[3][1] else 0)
    else:
        fingers.append(1 if landmarks[4][1] < landmarks[3][1] else 0)
    for tip_id in [8, 12, 16, 20]:
        fingers.append(1 if landmarks[tip_id][2] < landmarks[tip_id-2][2] else 0)
    is_fist = sum(fingers[1:]) == 0
    return fingers, is_fist

try:
    while True:
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        landmarks = get_landmarks(img, results)

        if landmarks and len(landmarks) >= 20:
            fingers, is_fist = check_fingers(landmarks)
            index_x, index_y = landmarks[8][1], landmarks[8][2]
            thumb_x, thumb_y = landmarks[4][1], landmarks[4][2]
            middle_tip = (landmarks[12][1], landmarks[12][2])
            current_time = time.time()

            if is_fist:
                if not dragging:
                    if current_time - drag_start_time > DRAG_ACTIVATION_TIME:
                        dragging = True
                        pyautogui.mouseDown()
                        drag_start_pos = (index_x, index_y)
                        cv2.putText(img, "DRAG START", (10, 200), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    else:
                        drag_start_time = current_time
                else:
                    screen_x = np.interp(index_x, (frameR, wCam-frameR), (0, pyautogui.size()[0]))
                    screen_y = np.interp(index_y, (frameR, hCam-frameR), (0, pyautogui.size()[1]))
                    curr_x = prev_x + (screen_x - prev_x) / smoothening
                    curr_y = prev_y + (screen_y - prev_y) / smoothening
                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y
                    cv2.rectangle(img, 
                                (drag_start_pos[0]-50, drag_start_pos[1]-50),
                                (drag_start_pos[0]+50, drag_start_pos[1]+50),
                                (255,165,0), 2)
            elif dragging:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(img, "DRAG END", (10, 200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            if fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
                if not scroll_active:
                    lock_mouse_pos = pyautogui.position()
                    prev_x, prev_y = lock_mouse_pos.x, lock_mouse_pos.y
                    scroll_start_y = (landmarks[8][2] + landmarks[12][2]) / 2
                    prev_scroll_time = current_time
                    scroll_active = True
                
                pyautogui.moveTo(lock_mouse_pos.x, lock_mouse_pos.y)
                current_y = (landmarks[8][2] + landmarks[12][2]) / 2
                time_diff = current_time - prev_scroll_time
                
                if time_diff > 0.05:
                    delta_y = scroll_start_y - current_y
                    velocity = delta_y / time_diff
                    
                    if abs(velocity) > MOVEMENT_THRESHOLD:
                        scroll_amount = int(velocity * SCROLL_SENSITIVITY)
                        pyautogui.scroll(scroll_amount)
                        
                        color = (0,255,0) if scroll_amount >0 else (0,0,255)
                        cv2.putText(img, f"SCROLL {'UP' if scroll_amount>0 else 'DOWN'}",
                                  (10,160), cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                        cv2.line(img, (wCam//2,hCam//2),
                                (wCam//2, hCam//2 - scroll_amount//2), color,3)

                    scroll_start_y = current_y
                    prev_scroll_time = current_time
            else:
                if scroll_active:
                    prev_x, prev_y = pyautogui.position()
                scroll_active = False

            if not dragging and not scroll_active and fingers[1] == 1 and sum(fingers) <= 3:
                screen_x = np.interp(index_x, (frameR, wCam-frameR), (0, pyautogui.size()[0]))
                screen_y = np.interp(index_y, (frameR, hCam-frameR), (0, pyautogui.size()[1]))
                
                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y
                cv2.circle(img, (index_x, index_y), 10, (255,0,255), cv2.FILLED)

            left_dist = np.hypot(index_x - thumb_x, index_y - thumb_y)
            if left_dist < click_distance and (current_time - last_click_time) > click_cooldown:
                pyautogui.click()
                last_click_time = current_time
                cv2.circle(img, (index_x, index_y), 10, (0,255,0), cv2.FILLED)
                cv2.circle(img, (thumb_x, thumb_y), 10, (0,255,0), cv2.FILLED)
                cv2.line(img, (index_x, index_y), (thumb_x, thumb_y), (0,255,0), 2)
                cv2.putText(img, "LEFT CLICK", (10,30), 
                          cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

            right_dist = np.hypot(middle_tip[0] - thumb_x, middle_tip[1] - thumb_y)
            if right_dist < click_distance and (current_time - last_click_time) > click_cooldown:
                pyautogui.rightClick()
                last_click_time = current_time
                cv2.circle(img, middle_tip,10,(0,0,255),cv2.FILLED)
                cv2.circle(img, (thumb_x,thumb_y),10,(0,0,255),cv2.FILLED)
                cv2.line(img, middle_tip, (thumb_x,thumb_y), (0,0,255),2)
                cv2.putText(img, "RIGHT CLICK", (10,60), 
                          cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)

        cv2.rectangle(img, (frameR,frameR), (wCam-frameR,hCam-frameR), (255,0,0),2)
        cv2.imshow("Virtual Mouse", img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()