import os
print(os.getcwd())
from VideoSegmentDataset import VideoSegmentDataset

dataset = VideoSegmentDataset(root_dir="/home/local/VANDERBILT/winterga/medic/data/images_ts", seq_length=10)
print(len(dataset))
frames, label = dataset[0]
print(frames.shape, label)
