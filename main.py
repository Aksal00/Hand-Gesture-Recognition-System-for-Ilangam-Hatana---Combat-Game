import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math

movement = "idle"
def cam_output(movement):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    def calculate_distance(point1, point2):
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    
        
    
                
                
    cap = cv2.VideoCapture(0)
     
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        while cap.isOpened():
            ret, frame = cap.read()
            
            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Flip on horizontal
            image = cv2.flip(image, 1)
            
            # Set flag
            image.flags.writeable = False
            
            # Detections
            results = hands.process(image)
            
            # Set flag to true
            image.flags.writeable = True
            
            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detections
            #print(results)
            
            cv2.putText(image, "ILANGAM HATANA HAND GESTURE TRACKER", (120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0 ,255 ), 2)
            # Rendering results
            movement = "idle"
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )
                    # Get landmarks of thumb and index finger
                    index_pip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                    index_tip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                    thumb_tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    thumb_mcp = hand.landmark[mp_hands.HandLandmark.THUMB_MCP]
                    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]
                    middle_pip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                    middle_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_pip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                    ring_tip = hand.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_pip = hand.landmark[mp_hands.HandLandmark.PINKY_PIP]
                    pinky_tip = hand.landmark[mp_hands.HandLandmark.PINKY_TIP]
                    # Calculate distance between thumb and index finger tips
                    
                    thumb_index_distance = calculate_distance(thumb_tip, index_mcp)
                  #Identifying the right hand
                    if wrist.x >= 0.5:
                        if (index_pip.y >index_tip.y) and (middle_pip.y > middle_tip.y) and (ring_pip.y > ring_tip.y) and (pinky_pip.y > pinky_tip.y)and thumb_index_distance >= 0.12:
                            cv2.putText(image, "idle", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 255), 2)
                            right_hand_movement = "idle"

                        elif (index_pip.y <index_tip.y) and (middle_pip.y < middle_tip.y) and (ring_pip.y < ring_tip.y) and (pinky_pip.y < pinky_tip.y)and thumb_index_distance < 0.12 :
                            cv2.putText(image, "shield", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            right_hand_movement = "shield"  
                        elif wrist.y < 0.5:
                            if thumb_index_distance >= 0.12:
                                cv2.putText(image, "attack_1", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                right_hand_movement = "attack_1"  
                        elif wrist.y > 0.85:
                            if thumb_index_distance >= 0.12:
                                cv2.putText(image, "attack_2", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                right_hand_movement = "attack_2"
                        
                        elif 0.5 <= wrist.y <= 0.85:
                            if thumb_index_distance >= 0.12:
                                cv2.putText(image, "attack_3", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                right_hand_movement = "attack_3"
                            
                    #Identifying the left hand
                    if wrist.x < 0.5:
                        
                        if thumb_index_distance >= 0.09:
                            #cv2.putText(image, "Left Thumb Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            if thumb_tip.x > wrist.x:
                                cv2.putText(image, "move right", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                
                                left_hand_movement = "move_right"
                            else:
                                cv2.putText(image, "move left", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                left_hand_movement = "move_left"
                        
                        
                        if (index_pip.y >index_tip.y) and (middle_pip.y > middle_tip.y) and (ring_pip.y > ring_tip.y) and (pinky_pip.y > pinky_tip.y):
                            cv2.putText(image, "jump", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            left_hand_movement = "jump"

                        elif thumb_index_distance <0.09:
                            cv2.putText(image, "idle", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 255), 2)
                            left_hand_movement = "idle"
                    
                    
            else:
                left_hand_movement = "idle"
                right_hand_movement = "idle"
                cv2.putText(image, "idle", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.putText(image, "idle", (520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 255), 2)
            print ("Left Hand :- "+left_hand_movement+"   Right Hand :- "+right_hand_movement)
            
            #left_hand_gesture_queue.put(left_hand_movement)
            #right_hand_gesture_queue.put(right_hand_movement)
            cv2.imshow('Ilangam Hatana - Hand Gesture Tracker', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            #cap.release()
            #cv2.destroyAllWindows()
                
               
     
            

            

        

    
        

    
    
while True:
    print (movement)
    movement = cam_output(movement)
    
    #cap.release()
cv2.destroyAllWindows()