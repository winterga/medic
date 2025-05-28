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
    saved_ids = set()  # Track unique frames: (video_name, frame_number)

    for video in video_list:
        padding = detect_padding(os.path.join('../images_ts_vids', video), video)
        print(f"Processing {split_name}: {video}")

        for index in video_list[video]:
            start = index - 30
            end = index + 30
            print(f"Range: {start} to {end}")

            for i in range(start, end + 1):
                frame_id = (video, i)
                if frame_id in saved_ids:
                    continue  # Already copied this frame

                frame_number = str(i).zfill(padding)
                frame_filename = os.path.join(video, f'_frame_{frame_number}').replace('/', '') + '.jpg'
                frame_path = os.path.join('../images_ts_vids', video, frame_filename)

                if not os.path.exists(frame_path):
                    print(f"⚠️ Frame not found: {frame_path}")
                    continue

                destination_folder = os.path.join('../images_ts_fe_30_singles', split_name)
                os.makedirs(destination_folder, exist_ok=True)
                shutil.copy(frame_path, destination_folder)
                saved_ids.add(frame_id)

                print(f"✅ Copied {frame_filename} to {destination_folder}")


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
            (7984), (8635)
        ],
        "09_30_2022_11_38_00_fullvid.MP4": [
            (742), (52674), (53048), (53082), (53119), (53728) #'''(53099, 53112)''',
        ],
        "02_02_2024_08_41_58_fullvid.mp4": [
            (20982), (21098), (21229), (22026), (22327), (24008), (24873)
        ],
        "01_30_2023_07_11_48_fullvideo.MP4": [
            (58365)
        ],
        "12_22_2022_06_48_13_fullvid.MP4": [
            (18846), (19463), (34249), (37824), (38806), (40875), (44452), (52698), (53242), (55451), (59200), (59488)#, (60366, 60379)
        ]
    }

    valid_list = {
        "10_31_2022_07_18_14_fullvid.MP4": [
            (505)
        ]
    }

    test_list = {
        "10_31_2023_14_36_44_fullvideo.mp4": [
            (4585), (5241), (7432), (8230), (9950), (10881), (12600), (13725), (14870)
        ],
        "08_22_2022_12_54_56_fullvid.MP4": [
            (78453), (78950), (78991), (79021)
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
    process_video_splits(valid_list, "val")
    process_video_splits(test_list, "test")
                

    print("✅ Frame extraction complete.")
