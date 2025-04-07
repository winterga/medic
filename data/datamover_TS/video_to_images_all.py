import os
import cv2  # OpenCV to read video lengths
import argparse
import time
import glob
import shutil

def get_video_duration(video_path):
    """Returns the duration of the video in seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count / fps if fps > 0 else 0

def save_frames(video, output_dir):
    """Extract frames from the video and saves them in a subdirectory named after the video."""
    video_name = os.path.basename(video)
    cap = cv2.VideoCapture(video)

    # Create a folder for the video
    video_folder = os.path.join(output_dir, video_name)
    os.makedirs(video_folder, exist_ok=True)

    frame_count = 0
    # Get the number of digits to pad the frame numbers for natural sorting
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start from the beginning
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    digits = len(str(total_frames))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Generate a zero-padded frame filename based on the total number of frames
        frame_filename = f'{video_name}_frame_{str(frame_count).zfill(digits)}.jpg'
        frame_path = os.path.join(video_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"✅ Processed {frame_count} frames for {video_name}")

def detect_padding(video_path, video):
    frame_files = glob.glob(os.path.join(video_path, video + "_frame_*"))
    print(frame_files[0])
    if not frame_files:
        return 5  # Default to 5 if no files are found
    example_frame = os.path.basename(frame_files[0])
    print('T', example_frame)
    digits = len(example_frame.split("_frame_")[1])-4 # 4 is the length of the string ".jpg"
    print("Digits", digits)
    return digits

def process_video_splits(video_list, split_name):
    for video in video_list:
        padding = detect_padding(os.path.join('../images_ts_vids', video), video)
        print(f"Processing {split_name}: {video}")
        
        for start, end in video_list[video]:
            far_left = start - 20
            far_right = end + 20
            print(f"Frames: {far_left} - {far_right}")

            for i in range(far_left, far_right):
                frame_number = str(i).zfill(padding)
                frame_path = os.path.join('../images_ts_vids', video, os.path.join(video, f'_frame_{frame_number}').replace('/', '')) + '.jpg'
                
                destination_folder = None
                if i < start:
                    destination_folder = os.path.join('../images_ts', split_name, '0')  # Pre-event (class 0)
                elif start <= i <= end:
                    destination_folder = os.path.join('../images_ts', split_name, '1')  # Event (class 1)
                elif i > end:
                    destination_folder = os.path.join('../images_ts', split_name, '0')  # Post-event (class 0)
                
                if destination_folder:
                    os.makedirs(destination_folder, exist_ok=True)
                    shutil.copy(frame_path, destination_folder)
                else:
                    raise SystemError("Unexpected case encountered!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from specific videos.')
    parser.add_argument('--videos_dir', type=str, required=True, help='Path to the directory containing videos.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory for frames.')
    args = parser.parse_args()

    # Define the list of video filenames to process
    video_list = [
        "01_30_2023_07_11_48_fullvideo.MP4",
        "02_02_2024_08_41_58_fullvid.mp4",
        "02_05_2021_16_32_47_fullvideo.mp4",
        "08_22_2022_12_54_56_fullvid.MP4",
        # "09_11_2023_14_35_35_fullvid.MP4",
        "09_30_2022_11_38_00_fullvid.MP4",
        "10_31_2022_07_18_14_fullvid.MP4",
        "10_31_2023_14_36_44_fullvideo.mp4",
        "11_07_2023_10_58_44_Ch2_001_CH001_V.MP4",
        "12_22_2022_06_48_13_fullvid.MP4"
        # "09_11_2023_14_35_35_fullvid.MP4"
    ]   

    train_list = {
        "02_05_2021_16_32_47_fullvideo.mp4": [
            (7983, 7999), (8630, 8646)
        ],
        "09_30_2022_11_38_00_fullvid.MP4": [
            (736, 749), (52670, 52683), (53047, 53060), (53078, 53091), (53099, 53112), (53116, 53129), (53721, 53734)
        ],
        "02_02_2024_08_41_58_fullvid.mp4": [
            (20980, 20993), (21098, 21111), (21215, 21228), (22017, 22030), (22322, 22335), (24002, 24015), (24865, 24878)
        ],
        "01_30_2023_07_11_48_fullvideo.MP4": [
            (58363, 58376)
        ],
        "11_07_2023_10_58_44_Ch2_001_CH001_V.MP4": [
            (7419, 7432), (7440, 7453), (9736, 9749)
        ],
        "12_22_2022_06_48_13_fullvid.MP4": [
            (18841, 18854), (19461, 19474), (34237, 34250), (37815, 37828), (38793, 38806), (40867, 40880), (43278, 43291), (44443, 44456), (52688, 52701), (53227, 53240), (55432, 55445), (59185, 59198), (59485, 59498), (60366, 60379)
        ]
    }

    valid_list = {
        "10_31_2022_07_18_14_fullvid.MP4": [
            (503, 516)
        ]
    }

    test_list = {
        "10_31_2023_14_36_44_fullvideo.mp4": [
            (4583, 4596), (5238, 5251), (7431, 7444), (8219, 8232), (9947, 9960), (10878, 10891), (12578, 12611), (13722, 13735), (14867, 14880)
        ],
        "08_22_2022_12_54_56_fullvid.MP4": [
            (78441, 78454), (789943, 78956), (78989, 79002), (79018, 79031)
        ]
    }

    video_directory = args.videos_dir
    videos = [f for f in os.listdir(video_directory) if os.path.isfile(os.path.join(video_directory, f)) and f in video_list]

    print(f"📂 Found {len(videos)} matching videos")

    # for video in videos:
    #     video_path = os.path.join(video_directory, video)
    #     duration = get_video_duration(video_path)
    #     print(f"⏳ {video} duration: {duration:.2f} seconds")
    #     save_frames(video_path, args.output_dir)

    # Process each dataset split
    process_video_splits(train_list, "train")
    process_video_splits(valid_list, "valid")
    process_video_splits(test_list, "test")
                

    print("✅ Frame extraction complete.")
