import cv2 as cv
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

hand_model_path = "hand_landmarker.task"
pose_model_path = "pose_landmarker_full.task"
face_model_path = "face_landmarker.task"

# Convert landmark list to NormalizedLandmarkList
def convert_to_proto(landmarks):
    """Convert a list of landmarks into a NormalizedLandmarkList."""
    proto = landmark_pb2.NormalizedLandmarkList()
    for landmark in landmarks:
        proto.landmark.add(x=landmark.x, y=landmark.y, z=landmark.z)
    return proto

# Main Loop
def main():
    # Initialize Hand Landmarker
    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    hand_landmarker = HandLandmarker.create_from_options(hand_options)

    # Initialize Pose Landmarker
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    # Initialize Face Landmarker
    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.IMAGE,
    )
    face_landmarker = FaceLandmarker.create_from_options(face_options)

    webcam = cv.VideoCapture(0)

    while True:
        isTrue, frame = webcam.read()
        if isTrue:
            # Convert to RGB
            RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            media_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB_frame)

            # Perform detections
            hand_results = hand_landmarker.detect(media_image)
            pose_results = pose_landmarker.detect(media_image)
            face_results = face_landmarker.detect(media_image)

            # Annotate the frame
            annotated_image = frame.copy()

            # Draw hand landmarks (for multiple hands)
            if hand_results.hand_landmarks:
                for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    landmarks_proto = convert_to_proto(hand_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        landmarks_proto,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )
                    # Debug handedness structure
                    print("Handedness:", hand_results.handedness)

                    # Extract handedness
                    handedness_info = hand_results.handedness[i]
                    label = handedness_info[0].category_name  # Assuming it's a list of objects with category_name

                    # Label the hand (Left or Right)
                    x = int(landmarks_proto.landmark[0].x * frame.shape[1])
                    y = int(landmarks_proto.landmark[0].y * frame.shape[0])
                    cv.putText(annotated_image, label, (x, y - 20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Draw pose landmarks
            if pose_results.pose_landmarks:
                for pose_landmarks in pose_results.pose_landmarks:
                    landmarks_proto = convert_to_proto(pose_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        landmarks_proto,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2),
                    )

            # Draw face landmarks
            if face_results.face_landmarks:
                for face_landmarks in face_results.face_landmarks:
                    landmarks_proto = convert_to_proto(face_landmarks)
                    mp.solutions.drawing_utils.draw_landmarks(
                        annotated_image,
                        landmarks_proto,
                        mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        mp.solutions.drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                        mp.solutions.drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
                    )

            # Show the annotated frame
            cv.imshow("Webcam", annotated_image)

        # Exit on ESC key
        if cv.waitKey(1) & 0xFF == 27:
            break

    webcam.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
