import os
import cv2  # OpenCV to read video lengths
import random
import shutil
import math
import argparse
from video_splits import video_splits

def get_video_length(video_path):
    """Returns the length of the video in seconds."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    length = int(frame_count / fps) if fps > 0 else 0
    capture.release()
    return length

def save_frames(video, video_splits, output_dir):
    """Extract frames from the video and saves them in class folders (0, 1, 2, 3)."""
    video_name = os.path.basename(video)
    cap = cv2.VideoCapture(video)

    # Print the intervals for the video once
    print(f"Processing video: {video_name}")
    video_intervals = video_splits.get(video_name, {})
    print("Intervals:", video_intervals)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)  # Convert frame index to seconds
        
        # For each class, check if the timestamp falls within any interval
        for class_id, intervals in video_intervals.items():
            class_folder = os.path.join(output_dir, f'{class_id}')
            os.makedirs(class_folder, exist_ok=True)

            for interval in intervals:
                if interval and interval[0] <= timestamp < interval[1]:  # Check if the timestamp is within the interval
                    print(f"🖼️ Saving frame {frame_count} for class {class_id} (timestamp: {timestamp})")
                    frame_path = os.path.join(class_folder, f'{video_name}_frame_{frame_count}.jpg')
                    cv2.imwrite(frame_path, frame)
                    break  # Save the frame only for the first matching class interval
        
        frame_count += 1
    cap.release()

def split_frames(output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Splits frames from each class (0, 1, 2, 3) into train, val, and test sets."""
    
    # Dictionary to hold frame lists for each class
    all_frames = {0: [], 1: [], 2: [], 3: []}

    # Collect frames for each class
    for class_id in range(len(all_frames)):  # Since we have three classes (0, 1, 2, 3)
        class_folder_path = os.path.join(output_dir, f'{class_id}')
        if os.path.isdir(class_folder_path):
            all_frames[class_id] = [f for f in os.listdir(class_folder_path) if os.path.isfile(os.path.join(class_folder_path, f))]

    # Shuffle the frames for randomness
    for class_id in range(len(all_frames)):
        random.shuffle(all_frames[class_id])

    # Split each class into train, val, and test
    split_data = {'train': {}, 'val': {}, 'test': {}}
    for class_id in range(len(all_frames)):
        total = len(all_frames[class_id])
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)

        split_data['train'][class_id] = all_frames[class_id][:train_end]
        split_data['val'][class_id] = all_frames[class_id][train_end:val_end]
        split_data['test'][class_id] = all_frames[class_id][val_end:]

    # Move frames into train, val, test directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_id in range(len(all_frames)):
            class_split_dir = os.path.join(split_dir, str(class_id))
            os.makedirs(class_split_dir, exist_ok=True)

            for frame in split_data[split][class_id]:
                src_path = os.path.join(output_dir, f'{class_id}', frame)
                dst_path = os.path.join(class_split_dir, frame)
                shutil.move(src_path, dst_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run experiment with different hyperparameters.')
    parser.add_argument('--videos_dir', type=str, default='/home/user/Desktop/Research/Raw Data/residuals/Square', help='Path to the directory containing videos.')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Research/Raw Data/test', help='Path to the output directory for frames.')
    args = parser.parse_args()

    video_directory = args.videos_dir
    videos = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if os.path.isfile(os.path.join(video_directory, f))]
    
    print(f"📂 Found {len(videos)} videos")

    # Step 1: Extract frames from videos and save them in class folders
    for video in os.listdir(args.videos_dir):
        video_path = os.path.join(args.videos_dir, video)
        if os.path.isfile(video_path):
            save_frames(video_path, video_splits, args.output_dir)

    # Step 2: Split frames into train, val, and test sets
    split_frames(args.output_dir)

    print("✅ Frame extraction complete.")