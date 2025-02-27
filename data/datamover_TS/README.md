# DataMover

1. Create a `videos` and `images_ts` folder inside the `data` directory.
2. Put the videos in the `videos` folder.
3. Go to the `datamover/` directory
4. Execute `python video_to_images.py --videos_dir ../videos --output_dir ../images_ts`
5. When complete, execute `chmod +777 datamover.sh` if not done so already
6. Execute `./datamover.sh ../images_ts`