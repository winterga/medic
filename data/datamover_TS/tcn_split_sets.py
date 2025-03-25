import os
import shutil
import random
import argparse
from concurrent.futures import ThreadPoolExecutor

def copy_frame(src_path, dst_path):
    shutil.copy(src_path, dst_path)

def split_frames(output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    video_class_mapping = {}
    class_video_groups = {}

    # Step 1: Collect video-class mappings
    print(f"\n📂 **Starting collection of video-class mappings** from directory: {output_dir}")

    for video_name in os.listdir(output_dir):
        video_path = os.path.join(output_dir, video_name)
        print(f"🔍 Checking video: {video_name}")
        if not os.path.isdir(video_path):
            continue  # Skip non-folder files

        video_class_mapping[video_name] = {}

        for class_id in os.listdir(video_path):
            print(f"🔍 Checking class: {class_id}")
            class_folder = os.path.join(video_path, class_id)

            if not os.path.isdir(class_folder):
                continue  # Skip non-class folders

            frame_paths = [f.name for f in os.scandir(class_folder) if f.is_file()]
            num_frames = len(frame_paths)

            if num_frames > 0:
                video_class_mapping[video_name][class_id] = frame_paths
                if class_id not in class_video_groups:
                    class_video_groups[class_id] = []
                class_video_groups[class_id].append((video_name, num_frames))
                print(f"🔄 Processing video: {video_name}")
                print(f"    - Class {class_id}: Found {num_frames} frames")

    # Step 2: Sort videos within each class by frame count (largest → smallest)
    print("\n📝 Sorting videos within each class by frame count (largest → smallest)...")

    for class_id in class_video_groups:
        class_video_groups[class_id].sort(key=lambda x: x[1], reverse=True)

    # Step 3: Assign videos to Train, Val, Test
    print("\n📊 **Assigning videos to Train, Val, and Test splits...**")
    split_assignments = {'train': set(), 'val': set(), 'test': set()}
    remaining_videos = set(video_class_mapping.keys())

    # Step 3a: Ensure every split contains at least one video per class
    for class_id, video_list in class_video_groups.items():
        for split in ['train', 'val', 'test']:
            valid_video_list = [video for video, _ in video_list if class_id in video_class_mapping[video]]
            
            # Ensure that a video containing the class is added to each split
            for video_name in valid_video_list:
                if video_name in remaining_videos:
                    split_assignments[split].add(video_name)
                    remaining_videos.remove(video_name)
                    break  # Ensuring class is represented in the split

    print(f"\n✅ Ensured that each split contains at least one video per class.")


    # Step 3b: Distribute remaining videos by ratio
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

    print(f"\n✅ Distributed remaining videos by ratio: {train_ratio}/{val_ratio}/{test_ratio}")

    # Step 4: Move images into corresponding train/val/test directories
    print("\n📦 **Moving images into corresponding train/val/test directories...**")
    for split, videos in split_assignments.items():
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        for class_id in {0, 1, 2}:  # Ensure class folders exist
            os.makedirs(os.path.join(split_dir, str(class_id)), exist_ok=True)

        for video_name in videos:
            for class_id, frame_paths in video_class_mapping[video_name].items():
                class_folder = os.path.join(split_dir, str(class_id))

                video_class_folder = os.path.join(output_dir, video_name, str(class_id))
                for frame_name in frame_paths:
                    src_path = os.path.join(video_class_folder, frame_name)
                    dst_path = os.path.join(class_folder, frame_name)
                    shutil.copy(src_path, dst_path)

    # Step 5: Remove original video folders
    print("\n🗑 **Removing original video folders...**")
    for video_name in video_class_mapping.keys():
        video_folder = os.path.join(output_dir, video_name)
        if os.path.isdir(video_folder):
            shutil.rmtree(video_folder)

    # Step 6: Debugging - Print split distributions
    print("\n📊 **Final Split Assignments:**")
    for split in ['train', 'val', 'test']:
        print(f"\n🔹 **{split.upper()}**")
        for class_id in {0, 1, 2}:
            class_folder = os.path.join(output_dir, split, str(class_id))
            num_files = len(os.listdir(class_folder)) if os.path.exists(class_folder) else 0
            print(f"   - Class {class_id}: {num_files} images")

    return split_assignments


if __name__ == '__main__':
    num_cores = os.cpu_count()
    parser = argparse.ArgumentParser(description='Split frames into train, val, and test sets')
    parser.add_argument('--base_dir', type=str, default='/home/local/VANDERBILT/winterga/medic/data', help='Base directory containing train, val, and test directories')
    parser.add_argument('--output_dir', type=str, default='/home/local/VANDERBILT/winterga/medic/data/images_ts', help='Output directory for extracted frames')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of data to use for validation')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of data to use for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for shuffling data')
    args = parser.parse_args()

    assert args.train_ratio + args.val_ratio + args.test_ratio == 1.0, "Ratios must sum to 1.0"

    videos = []


    split_frames(args.output_dir, args.train_ratio, args.val_ratio, args.test_ratio)
