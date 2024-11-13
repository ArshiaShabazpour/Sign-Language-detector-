import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = r'data\Landmark_data.csv'
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, 65)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=42,shuffle=True)
model = SVC(probability=True)
print(X_train[0].size)
model.fit(X_train,y_train)

y_predct = model.predict(X_test)

score = accuracy_score(y_predct,y_test)
print(score)



def draw_landmarks_on_image(rgb_image, detection_result,font_size=1,font_thickness=1,handness_color=(225,225,225),handness_border_color = (0,0,0),margin=15):
  annotated_image = rgb_image
  hand_landmarks_list = detection_result.hand_landmarks
  hand_world_landmarks_list = detection_result.hand_world_landmarks
  handedness_list = detection_result.handedness

  for i in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[i]
    handedness = handedness_list[i]

    hand_landmarks_normalized = landmark_pb2.NormalizedLandmarkList()
    for landmark in hand_landmarks:
        normalized_landmark = landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
        hand_landmarks_normalized.landmark.extend([normalized_landmark])
        
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_normalized,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())
    
  annotated_image = cv.flip(annotated_image,1)

  for i in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[i]
    handedness = handedness_list[i]
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(annotated_image.shape[1]- (max(x_coordinates) * annotated_image.shape[1])) 
    text_y = int(min(y_coordinates) * annotated_image.shape[0]) - margin

    cv.putText(annotated_image, handedness[0].display_name, (text_x, text_y), cv.FONT_HERSHEY_TRIPLEX, font_size, handness_border_color, font_thickness+2)
    cv.putText(annotated_image, handedness[0].display_name, (text_x, text_y), cv.FONT_HERSHEY_TRIPLEX,font_size, handness_color, font_thickness)
    
  return annotated_image

def draw_detection_box(results,image,character):
   annotated_image = cv.flip(image,1)
   hand_landmarks_list = results.hand_landmarks
   for i in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[i]
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    x_top_right = int(annotated_image.shape[1]- (max(x_coordinates) * annotated_image.shape[1])) - 25
    y_top_right = int(min(y_coordinates) * annotated_image.shape[0]) - 25
    x_bottom_left = int(annotated_image.shape[1]- (min(x_coordinates) * annotated_image.shape[1])) + 25
    y_bottom_left = int(max(y_coordinates) * annotated_image.shape[0]) + 25
    cv.rectangle(annotated_image,(x_top_right,y_top_right),(x_bottom_left,y_bottom_left),(0,225,0),thickness=10) 
    cv.rectangle(annotated_image,(x_top_right,y_top_right),(x_bottom_left,y_bottom_left),(0,0,0),thickness=3)
    cv.rectangle(annotated_image,(x_top_right-3,y_top_right-20),(((x_bottom_left+x_top_right)//2) -20,y_top_right),(0,0,0),thickness=-1)
    cv.putText(annotated_image,character[i],(x_top_right+5,y_top_right),cv.FONT_HERSHEY_TRIPLEX,color=(0,225,0),fontScale=0.75)

    
   return annotated_image
def save_as_csv(list_dictionary):
    data_frame = pd.DataFrame(list_dictionary)
    parent_directory = os.path.dirname(os.path.realpath(__file__))
    leaf_directory = os.path.join(parent_directory, f"data")

    if not os.path.exists(leaf_directory):
        os.makedirs(leaf_directory)

    file_path = os.path.join(leaf_directory, f"Landmark_data.csv")

    if os.path.exists(leaf_directory):
        data_frame.to_csv(file_path, mode='a', header=False, index=False)
    else:
        data_frame.to_csv(file_path, mode='a', header=True, index=False)

def add_to_list(results, list_dictionary, letter):
    hand_landmarks_list = results.hand_landmarks
    handedness_list = results.handedness
    for i in range(len(hand_landmarks_list)):
       handedness = 1 if handedness_list[i][0].display_name == "Right" else 0
       dict_to_add ={"letter": ord(letter)-97,"handedness" :handedness }
       for j in range (len(hand_landmarks_list[i])):
        dict_to_add.update({f"x{j+1}": hand_landmarks_list[i][j].x,f"y{j+1}":hand_landmarks_list[i][j].y,f"z{j+1}": hand_landmarks_list[i][j].z})
       list_dictionary.append(dict_to_add)
    return list_dictionary

def remove_latest_from_list(list_dictionary):
   list_dictionary.pop()
   return list_dictionary

def train_model():
   
   return None

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_tracking_confidence	=0.5,
    min_hand_detection_confidence = 0.5
)
detector = vision.HandLandmarker.create_from_options(options)
webcam = cv.VideoCapture(0)

mode = 0 
letter = None
test = []
duplicate_switch = False

while True:
    isTrue, frame = webcam.read()
    key = cv.waitKey(4)

    if isTrue:
        RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_frame)
        results = detector.detect(media_image)

        if mode == 0:   #mode selection 
            annonated_picture = draw_landmarks_on_image(frame,results)
            cv.putText(annonated_picture,"Mode Selection",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture,"Mode Selection",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.putText(annonated_picture," . press 1 for capture mode",(20,40),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press 1 for capture mode",(20,40),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.putText(annonated_picture," . press 2 for detection mode",(20,55),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press 2 for detection mode",(20,55),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.putText(annonated_picture," . press \"esc\" for to close the app ",(20,70),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press \"esc\" for to close the app ",(20,70),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.imshow("webcam", annonated_picture)

            if key == 27 : 
               break
            if key == 49:
                mode = 1
            if key == 50:
              mode = 2 


        elif mode == 1:     #select letter mode 
            annonated_picture = draw_landmarks_on_image(frame,results)
            if key == 27 : 
               mode = 0
            if key > 96 and key < 123:
               letter = chr(key) 
               mode = 3
               key == None
            cv.putText(annonated_picture,f"please select the letter you want to sign",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture,f"please select the letter you want to sign",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.imshow("webcam", annonated_picture)

        elif mode == 3: #saving mode 
            annonated_picture = draw_landmarks_on_image(frame,results)
            if key == 27 : 
               test = []
               mode = 1
            if key == 32:
               duplicate_switch = True
               test = add_to_list(results,test,letter)
            if key == 115 and duplicate_switch == True:
               duplicate_switch = False
               save_as_csv(test)
               test = []
            if key == 122:
               list_dictionary = remove_latest_from_list(list_dictionary)

            cv.putText(annonated_picture,f"please do the sign for {letter}",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture,f"please do the sign for {letter}",(20,25),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)

            cv.putText(annonated_picture," . press \"z\" to reverse",(20,40),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press \"z\" to reverse",(20,40),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)

            cv.putText(annonated_picture," . press \"space\" to record data",(20,55),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press \"space\" to record data",(20,55),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)

            cv.putText(annonated_picture," . press \"s\" to save",(20,70),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press \"s\" to save",(20,70),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)

            cv.putText(annonated_picture," . press \"esc\" to go back",(20,85),cv.FONT_HERSHEY_COMPLEX,0.5,(0,0,0),3)
            cv.putText(annonated_picture," . press \"esc\" to go back",(20,85),cv.FONT_HERSHEY_COMPLEX,0.5,(225,225,225),1)
            cv.imshow("webcam", annonated_picture)
        elif mode == 2:
            predict_array = []
            prediction_results = []
            testing_value = None
            if key == 27 : 
               mode = 0
            if results.hand_landmarks:
                hand_landmarks_list = results.hand_landmarks
                handedness_list = results.handedness
                for i in range(len(hand_landmarks_list)):
                    handedness = handedness_list[i][0].display_name
                    handedness = 0 if handedness =="Left" else 1
                    predict_array.append(handedness)
                    for j in range (len(hand_landmarks_list[i])):
                        predict_array.append(hand_landmarks_list[i][j].x)
                        predict_array.append(hand_landmarks_list[i][j].y)
                        predict_array.append(hand_landmarks_list[i][j].z)
                    prediction =model.predict([predict_array])[0]+97
                    testing_value = max(model.predict_proba([predict_array])[0])
                    if testing_value < 0.8:
                       prediction = 63
                    print(testing_value)
                    predict_array = []
                    prediction_results.append(chr(prediction))
                annonated_picture = draw_detection_box(results,frame,prediction_results)
                prediction_results = []
            else:
               annonated_picture = cv.flip(frame,1)
            cv.putText(annonated_picture,f"detection mode",(20,25),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
            cv.putText(annonated_picture,f"detection mode",(20,25),cv.FONT_HERSHEY_COMPLEX,1,(225,225,225),1)
            cv.imshow("webcam", annonated_picture)
        
 

webcam.release()
cv.destroyAllWindows()
