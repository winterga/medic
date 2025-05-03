import sys
import os

import cv2
from src.crop import crop_and_resize
import csv
from src.video_splits import video_splits

sys.path.append('Documents/MEDIC')  
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_video_frames(videos, output_dir="extracted_frames"):
    for video in videos:
        print(video)
        video_name = os.path.basename(video)
        print(video_name)
        video_frame_dir = os.path.join(output_dir, video_name) 
        os.makedirs(video_frame_dir, exist_ok=True)

        cap = cv2.VideoCapture(video)
        print(f"Reading video: {video}")

        if not cap.isOpened():
            print("Error: Could not open video.")
            return video_name, None  

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop if no more frames

            frame = crop_and_resize(frame)
            frame_path = os.path.join(video_frame_dir, f"frame_{frame_count:05d}.jpg")  
            cv2.imwrite(frame_path, frame) 
            frame_count += 1
        cap.release()
        print(f"Extracted {frame_count} frames from {video} to {video_frame_dir}")

    return output_dir



def sort_videos(video_dir):
    splits = {k.strip().lower(): k for k in video_splits.keys()}
    print(f"Total video splits: {len(splits)}")

    video_paths = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    used_videos = []
    unused_videos = []

    for video in video_paths:
        video_clean = video.strip().lower()
        full_path = os.path.join(video_dir, video)

        if video_clean in splits:
            unused_videos.append(full_path)
        else:
            used_videos.append(full_path)

    print(f"✅ Used videos: {len(used_videos)}")
    print(f"❌ Unused videos: {len(unused_videos)}")

    return unused_videos




def save_frames(frames, labels, video_name, save_path):
    # Iterate over clusters and save features and frames
    for idx, label in enumerate(labels):
        label_dir = os.path.join(save_path, video_name, str(label)) 
        
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        frame = cv2.imread(frames[idx])
        
        if frame is None:
            print(f"Warning: Failed to load {frames[idx]}. Skipping.")
            continue

        frame_filename = os.path.join(label_dir, f"frame_{idx:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

    print(f"Spatial features and frames clustered and saved to {save_path}")

