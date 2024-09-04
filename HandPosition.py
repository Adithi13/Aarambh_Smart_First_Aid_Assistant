"""
In this we are detecting the correct hand posistions for the cpr


After getting the step by step instructions it will check the user, that the user is following 
step by step instruction based upon the predicted disease or symptoms

"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define the correct keypoint indices for hand and body positions
RIGHT_HAND = 20
LEFT_HAND = 19
RIGHT_SHOULDER = 12
LEFT_SHOULDER = 11
RIGHT_HIP = 24
LEFT_HIP = 23


def are_hand_positions_correct(keypoints):
    right_hand_y = keypoints[RIGHT_HAND].y if keypoints[RIGHT_HAND] else 0.0
    left_hand_y = keypoints[LEFT_HAND].y if keypoints[LEFT_HAND] else 0.0
    right_shoulder_y = keypoints[RIGHT_SHOULDER].y if keypoints[RIGHT_SHOULDER] else 0.0
    right_hip_y = keypoints[RIGHT_HIP].y if keypoints[RIGHT_HIP] else 0.0

    # Check if hands are above shoulders and below hips
    if right_hand_y < right_shoulder_y and left_hand_y < right_shoulder_y \
            and right_hand_y > right_hip_y and left_hand_y > right_hip_y:
        return True
    else:
        return False


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_height, frame_width, _ = frame.shape

        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect pose keypoints
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark
            keypoints = pose_landmarks  # Use the entire pose_landmarks list

            if are_hand_positions_correct(keypoints):
                cv2.putText(frame, "Correct CPR Hand Positions", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Incorrect Hand Positions", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("CPR Hand Position Guide", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
