import os
import cv2  # OpenCV to read video lengths
import random
import shutil
import math
import argparse
from video_splits import video_splits
import time

def get_video_length(video_path):
    """Returns the length of the video in seconds."""
    capture = cv2.VideoCapture(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    length = int(frame_count / fps) if fps > 0 else 0
    capture.release()
    return length

def save_frames(video, video_splits, output_dir):
    """Extract frames from the video and saves them in class folders (0, 1, 2)."""
    video_name = os.path.basename(video)
    cap = cv2.VideoCapture(video)

    # Print the intervals for the video once
    print(f"Processing video: {video_name}")
    video_intervals = video_splits.get(video_name, {})
    print("Intervals:", len(video_intervals))

    if len(video_intervals) == 0:
        print(f"⚠️ Skipping video {video_name} as it has no annotated intervals.")
        return
    # Create a video folder
    video_folder = os.path.join(output_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_count / cap.get(cv2.CAP_PROP_FPS)  # Convert frame index to seconds
        
        # For each class, check if the timestamp falls within any interval
        for class_id, intervals in video_intervals.items():
            class_folder = os.path.join(video_folder, f'{class_id}')
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
    """
    Splits videos into Train, Val, Test while ensuring:
    1. Videos are NOT split across multiple datasets.
    2. Each dataset contains at least one video with every class.
    3. Each dataset has **full class coverage** (i.e., all classes 0, 1, 2 are present).
    """

    video_class_mapping = {}  # {video_name: {class_id: num_frames}}
    class_video_groups = {}  # {class_id: [videos_containing_class]}

    start_1_time = time.time()
    # Step 1: Collect video-class mappings
    for video_name in os.listdir(output_dir):
        video_path = os.path.join(output_dir, video_name)
        if not os.path.isdir(video_path):
            continue  # Skip non-folder files

        video_class_mapping[video_name] = {}

        for class_id in os.listdir(video_path):
            class_folder = os.path.join(video_path, class_id)
            if not os.path.isdir(class_folder):
                continue  # Skip non-class folders

            frame_paths = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
            num_frames = len(frame_paths)

            if num_frames > 0:
                video_class_mapping[video_name][class_id] = frame_paths
                if class_id not in class_video_groups:
                    class_video_groups[class_id] = []
                class_video_groups[class_id].append((video_name, num_frames))
    end_1_time = time.time()
    print(f"Step 1: Collect video-class mappings took {end_1_time - start_1_time:.2f} seconds")

    # Step 2: Sort videos within each class by frame count (largest → smallest)
    start_2_time = time.time()
    for class_id in class_video_groups:
        class_video_groups[class_id].sort(key=lambda x: x[1], reverse=True)
    end_2_time = time.time()
    print(f"Step 2: Sort videos within each class took {end_2_time - start_2_time:.2f} seconds")

    # Step 3: Assign videos to Train, Val, Test
    start_3_time = time.time()
    split_assignments = {'train': set(), 'val': set(), 'test': set()}
    remaining_videos = set(video_class_mapping.keys())
    end_3_time = time.time()
    print(f"Step 3: Assign videos to Train, Val, Test took {end_3_time - start_3_time:.2f} seconds")

    # Step 3a: Ensure every split contains at least one video per class
    start_3a_time = time.time()
    for class_id, video_list in class_video_groups.items():
        for split in ['train', 'val', 'test']:
            valid_video_list = [video for video, _ in video_list if class_id in video_class_mapping[video]]
            
            # Ensure that a video containing the class is added to each split
            for video_name in valid_video_list:
                if video_name in remaining_videos:
                    split_assignments[split].add(video_name)
                    remaining_videos.remove(video_name)
                    break  # Ensuring class is represented in the split
    end_3a_time = time.time()
    print(f"Step 3a: Ensure every split contains at least one video per class took {end_3a_time - start_3a_time:.2f} seconds")

    # Step 3b: Ensure each split has **full class coverage**
    start_3b_time = time.time()
    class_2_videos = [video for video in video_class_mapping if "2" in video_class_mapping[video]]

    # Ensure class 2 is evenly distributed first
    split_targets = ['train', 'val', 'test']
    for i, video in enumerate(class_2_videos):
        split = split_targets[i % 2]  # Round-robin distribution
        split_assignments[split].add(video)
        remaining_videos.discard(video)

    # Now, distribute remaining videos to balance the dataset
    remaining_videos = list(remaining_videos)  # Convert to list for shuffling
    random.shuffle(remaining_videos)

    num_videos = len(video_class_mapping)
    train_count = int(train_ratio * num_videos)
    val_count = int(val_ratio * num_videos)
    test_count = num_videos - train_count - val_count

    # Add remaining videos to each split
    split_assignments['train'].update(remaining_videos[:train_count])
    split_assignments['val'].update(remaining_videos[train_count:train_count + val_count])
    split_assignments['test'].update(remaining_videos[train_count + val_count:])
    end_3b_time = time.time()
    print(f"Step 3b: Ensure each split has full class coverage took {end_3b_time - start_3b_time:.2f} seconds)")

    # Step 4: Move images into corresponding train/val/test directories
    start_4_time = time.time()
    for split, videos in split_assignments.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_id in {0, 1, 2}:  # Ensure class folders exist
            os.makedirs(os.path.join(split_dir, str(class_id)), exist_ok=True)

        for video_name in videos:
            for class_id, frame_paths in video_class_mapping[video_name].items():
                class_folder = os.path.join(split_dir, str(class_id))

                video_class_folder = os.path.join(output_dir, video_name, str(class_id))
                print(frame_paths)
                print(video_class_folder)
                for frame_name in frame_paths:
                    src_path = os.path.join(video_class_folder, frame_name)
                    dst_path = os.path.join(class_folder, frame_name)
                    # shutil.move(src_path, dst_path)
                    shutil.copy(src_path, dst_path)
    end_4_time = time.time()
    print(f"Step 4: Move images into corresponding train/val/test directories took {end_4_time - start_4_time:.2f} seconds")

    # Step 5: Remove original video folders
    start_5_time = time.time()
    for video_name in video_class_mapping.keys():
        video_folder = os.path.join(output_dir, video_name)
        if os.path.isdir(video_folder):
            shutil.rmtree(video_folder)
    end_5_time = time.time()
    print(f"Step 5: Remove original video folders took {end_5_time - start_5_time:.2f} seconds")

    # Step 6: Debugging - Print split distributions
    print("\n📊 **Final Split Assignments:**")
    for split in ['train', 'val', 'test']:
        print(f"\n🔹 **{split.upper()}**")
        for class_id in {0, 1, 2}:
            class_folder = os.path.join(output_dir, split, str(class_id))
            num_files = len(os.listdir(class_folder)) if os.path.exists(class_folder) else 0
            print(f"   - Class {class_id}: {num_files} images")

    return split_assignments

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run experiment with different hyperparameters.')
    parser.add_argument('--videos_dir', type=str, default='/home/user/Desktop/Research/Raw Data/residuals/Square', help='Path to the directory containing videos.')
    parser.add_argument('--output_dir', type=str, default='/home/user/Desktop/Research/Raw Data/test', help='Path to the output directory for frames.')
    args = parser.parse_args()

    video_directory = args.videos_dir
    videos = [os.path.join(video_directory, f) for f in os.listdir(video_directory) if os.path.isfile(os.path.join(video_directory, f))]
    
    print(f"📂 Found {len(videos)} videos")

    # IF the frames have already been extracted, comment this out. 
    # Step 1: Extract frames from videos and save them in class folders
    # for video in os.listdir(args.videos_dir):
    #     video_path = os.path.join(args.videos_dir, video)
    #     if os.path.isfile(video_path):
    #         save_frames(video_path, video_splits, args.output_dir)

    # Step 2: Split frames into train, val, and test sets
    split_frames(args.output_dir)

    print("✅ Frame extraction complete.")