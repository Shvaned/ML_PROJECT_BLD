import cv2
import mediapipe as mp
import numpy as np
import joblib
import subprocess
import time
import matplotlib.pyplot as plt

# Load the trained model and fitted scaler
model = joblib.load('video_cluster_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the previously fitted scaler

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose

# Class descriptions corresponding to the indices
class_descriptions = {
    0: "The target is being observant or focused on a task",
    1: "The target seems to be amongst a group of associates",
    2: "The target seems to be exhibiting a sense of urgency",
    3: "The target is showing signs of composure"
}

def extract_pose_features_with_visualization(video_path=None, is_realtime=False):
    """Extract pose landmarks from a video or webcam feed and visualize them."""
    cap = cv2.VideoCapture(video_path if not is_realtime else 0)  # Use webcam if realtime, else video file
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    features_list = []
    expected_feature_size = 33 * 4  # Pose landmarks (33 landmarks * 4 features per landmark)
    start_time = time.time()  # Start time to track real-time processing

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe Pose
        results = pose.process(rgb_frame)

        # Extract pose landmarks (33 landmarks * 4 features per landmark)
        landmarks = []
        if results.pose_landmarks:
            landmarks.extend(
                np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]).flatten())

            # Visualize pose landmarks
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # If no pose landmarks, fill with zeros
        if len(landmarks) == 0:
            landmarks = np.zeros(expected_feature_size)

        features_list.append(landmarks)

        # Display the frame
        cv2.imshow('Pose Visualization', frame)

        # Exit if 'q' is pressed or after 10 seconds for real-time
        if cv2.waitKey(1) & 0xFF == ord('q') or (is_realtime and time.time() - start_time > 10):
            break

    cap.release()
    pose.close()
    cv2.destroyAllWindows()

    return np.array(features_list)

# Ask if the user wants to do real-time processing
mode = input("Do you want to do real-time processing (y/n)? ").strip().lower()

if mode == 'y':
    # Real-time processing
    features = extract_pose_features_with_visualization(is_realtime=True)
else:
    # Input video processing
    video_path = input("Enter video path: ")  # Replace with your video path
    features = extract_pose_features_with_visualization(video_path=video_path)

# Ensure features are not empty
if features.size == 0:
    raise ValueError("No valid features were extracted from the video.")

# Summarize features (mean across frames)
summarized_features = np.mean(features, axis=0).reshape(1, -1)

# Standardize features using the scaler trained on 132 features
standardized_features = scaler.transform(summarized_features)  # Use the previously fitted scaler

# Predict the probabilities for all classes
class_probabilities = model.predict_proba(standardized_features)

# Get the class labels and the corresponding probabilities
class_labels = model.classes_  # The class labels from the trained model
predicted_class_probabilities = dict(zip(class_labels, class_probabilities[0]))

# Sort the class probabilities in the order of 0, 1, 2, 3
sorted_predictions = sorted(predicted_class_probabilities.items(), key=lambda item: item[0])

# Print the predicted classes with descriptions and percentages
print("The predicted classes and their probabilities for the video are:")
output_lines = []  # Store output lines to write to the file

for class_label, probability in sorted_predictions:
    description = class_descriptions.get(class_label, "No description available")
    probability_percentage = probability * 100
    print(f"{class_label}: {description} ({probability_percentage:.2f}%)")

    # Prepare the line to be written to the file
    output_lines.append(f"{probability_percentage:.2f}")

# Write predictions to classes.txt
with open('classes.txt', 'w') as file:
    file.write("\n".join(output_lines))

print("\n")

# Optionally, run another Python script after writing to the file
subprocess.run(["python", "final.py"])

# Plotting the classes and percentages as a bar chart
class_names = [class_descriptions[class_label] for class_label, _ in sorted_predictions]
probabilities = [probability * 100 for _, probability in sorted_predictions]

plt.figure(figsize=(10, 6))
plt.barh(class_names, probabilities, color='skyblue')
plt.xlabel('Probability (%)')
plt.title('Class Predictions and Probabilities')
plt.xlim(0, 100)

# Display the percentages on the bars
for index, value in enumerate(probabilities):
    plt.text(value + 2, index, f'{value:.2f}%', va='center')

# Show the plot
plt.show()
