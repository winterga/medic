from utils import extract_video_frames, sort_videos
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract, crop  and save frames")
    parser.add_argument("--video_dir", type=str, default="paths/to/all/videos/to/be/extracted", help="all video are contained in a sigle directory, enter the path to this directory")
    parser.add_argument("--video_split", type=str, default="path/to/video_split/python/file", help="path is a python file than vontains video split")
    parser.add_argument("--save_dir", type=str, default="path/to/save/all/extracted/frames/directories", help="enter path to save frames from video extraction")

    args = parser.parse_args()


    #sort video and store in directory
    required_videos= sort_videos(args.video_dir)
    video_dirs = extract_video_frames (required_videos, output_dir=args.save_dir)
