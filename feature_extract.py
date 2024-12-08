import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import time


def process_video_to_npy(video_path, output_dir="output_features"):
    import mediapipe as mp  # Move import here

    # Initialize Mediapipe Pose
    mp_pose = mp.solutions.pose

    # Extract video name and prepare output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}.npy")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Initialize Mediapipe Pose
        pose = mp_pose.Pose()

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        video_features = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to extract pose landmarks
            results = pose.process(rgb_frame)

            # Extract landmarks or store zeros if no landmarks are detected
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_features = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
            else:
                frame_features = np.zeros(33 * 4)  # 33 landmarks * 4 features

            video_features.append(frame_features)

        cap.release()
        pose.close()  # Ensure resources are freed

        # Save features as .npy
        video_features = np.array(video_features)
        np.save(output_path, video_features)
        return output_path

    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return None  # Return None for videos that failed


def process_wrapper(args):
    video_path, output_dir, start_time, processed_count, total_videos = args
    result = process_video_to_npy(video_path, output_dir)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    processed_count.value += 1
    print(
        f"Processed {processed_count.value}/{total_videos} videos. "
        f"Elapsed time: {elapsed_time:.2f}s"
    )
    return result


def main(video_list, output_dir="output_features", start_index=0, batch_size=100):
    if not video_list:
        print("No videos found to process.")
        return

    total_videos = len(video_list)
    video_list = video_list[start_index:]

    num_cores = min(cpu_count(), 8)  # Adjust the number of cores used
    print(f"Using {num_cores} cores for parallel processing.")

    from multiprocessing import Value, Manager
    processed_count = Manager().Value("i", start_index)

    start_time = time.time()

    # Process videos in batches
    for i in range(0, len(video_list), batch_size):
        batch = video_list[i:i+batch_size]
        args_list = [
            (video_path, output_dir, start_time, processed_count, total_videos)
            for video_path in batch
        ]

        # Process the current batch
        with Pool(num_cores) as pool:
            pool.map(process_wrapper, args_list)

        # Output progress for the batch
        total_time = time.time() - start_time
        print(f"Processed batch {i//batch_size + 1} of {len(video_list)//batch_size + 1}. Elapsed time: {total_time:.2f}s")



if __name__ == "__main__":
    # List all video files in the directory
    video_directory = "test"
    video_list = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if f.endswith(".mp4")]

    # Get the starting index from the user
    start_index = int(input("Enter the video number to start from (e.g., 1235 to start at the 1236th video): "))

    # Process all videos and save as .npy starting from the user input
    main(video_list, start_index=start_index)
