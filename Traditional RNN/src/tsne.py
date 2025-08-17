import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
# from .train import StrongTransitionLSTM  # assuming your model class is here
from .train import CNNModel, SequentialSequenceDataset   # assuming your CNN model class is here
import re
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from .truth_data import transition_frames
import torchvision.transforms as v2
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
import torch.nn as nn
import os
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict

class StrongTransitionLSTM(nn.Module):

    def __init__(self, feature_dim=6153, num_classes=3, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        feature_dim: Dimension of CNN feature output (2048 for ResNet-50's second-to-last layer)
        num_classes: Number of classes predicted by the CNN (3 for your 3 classes: 0, 1, 2)
        """
        super().__init__()
        
        # Update input_dim to the combined size (features + predictions)
        # input_dim = feature_dim + num_classes  # Combine CNN features and predictions
        
        # LSTM setup for sequence input (CNN features and predictions combined)
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer to output transition score
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        features, _ = self.lstm(x)             # [B, 5, 512]
        out = self.dropout(features)           # [B, 5, 512]
        out = self.fc(out)                     # [B, 5, 1]
        probs = torch.sigmoid(out).squeeze(-1) # [B, 5]
        return features, out, probs            # Keep time dimension


def natural_key(path):
    """
    Extracts numerical parts for natural sorting.
    E.g., 'frame_10.jpg' → 10
    """
    return extract_frame_number(path)  # You already have this function

class OrderedFrameDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = sorted(image_paths, key=natural_key)  # Use natural order
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, img_path
    
def get_relevant_frames_per_transition(image_paths, transition_windows):
    """
    Args:
        image_paths (List[str]): All available frame paths.
        transition_windows (Dict[str, List[Tuple[int, int, int]]]):
            A dictionary where each key is a video ID, and the value is a list of tuples:
            (transition_frame_number, frames_left, frames_right)

    Returns:
        List[str]: Filtered image paths based on the specified windows.
    """
    video_to_frames = defaultdict(list)
    frame_to_path = defaultdict(dict)

    # print(image_paths)
    print("Image Paths Len", len(image_paths))
    for path in image_paths:
        video_id = extract_video_id(path)
        frame_num = extract_frame_number(path)
        if(video_id not in video_to_frames ):
            print(f"Adding {video_id} from {path}")
            # Only add if the frame number is not already present for this video
            # This prevents duplicates in the video_to_frames mapping
        video_to_frames[video_id].append(frame_num)
        frame_to_path[video_id][frame_num] = path

    selected_paths = []

    # print("Transition Windows:", transition_windows)
    for video_id, transitions in transition_windows.items():
        if video_id not in video_to_frames:
            print(f"Warning: Video ID {video_id} not found in available frames.")
            continue

        available_frames = sorted(video_to_frames[video_id])

        for t_frame, left, right in transitions:
            start = t_frame - left
            end = t_frame + right
            selected = [f for f in available_frames if start <= f <= end]
            selected_paths.extend([frame_to_path[video_id][f] for f in selected if f in frame_to_path[video_id]])

    return selected_paths


# === CONFIG ===
# CKPT_PATH = "/home/user/Documents/GitHub/medic/Traditional RNN/checkpoints/Resnet50_070725_12/Resnet50_070725_12.pth"
CKPT_PATH = "/home/user/Documents/GitHub/medic/Traditional RNN/checkpoints/Resnet50_071625_07/Resnet50_071625_07.pth"
CNN_CKPT_PATH = "/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth"
# DATA_PATH = "/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/test"
DATA_PATH = "/home/user/Documents/GitHub/medic/data/images_ts/test"
DATA_PATH_fold1 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold1/test"
DATA_PATH_fold2 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold2/test"
DATA_PATH_fold3 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold3/test"
DATA_PATH_fold4 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold4/test"
DATA_PATH_fold5 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold5"
DATA_PATH_fold6 = "/home/user/Documents/GitHub/medic/data/images_ts/externals/folds/fold6"
INPUT_DIM = 6153

# === Setup device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load models ===
cnn_model = CNNModel().to(device)
cnn_model.eval()

lstm_model = StrongTransitionLSTM(INPUT_DIM).to(device)
lstm_model.load_state_dict(torch.load(CKPT_PATH))
lstm_model.eval()

# === Transforms ===
transform = v2.Compose([
    v2.Resize((512, 512)),
    v2.ToTensor(),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Dataset & Dataloader ===
dataset_sequences = SequentialSequenceDataset(DATA_PATH, transform=transform, transition_frames=transition_frames)
dataset_images = OrderedFrameDataset(DATA_PATH, transform=transform)
dataloader_sequences = DataLoader(dataset_sequences, batch_size=1, shuffle=False)
dataloader_images = DataLoader(dataset_images, batch_size=1, shuffle=False)

# === Collectors ===
all_embeddings = []
all_labels = []
cnn_tsne_features = []
cnn_tsne_labels = []
cnn_tsne_step_idx = []

# === Inference & Per-Sequence Visualizations ===
# Store global data
all_embeddings = []
all_labels = []
all_preds = []
color_map = []

# Per Sequence (5 timesteps) Analysis + Graphing
# ------------------------------------------------------------------------------

# with torch.no_grad():
#     for idx, (sequences, paths, label, info) in enumerate(dataloader_sequences):
#         B, num_parts, S, C, H, W = sequences.shape
#         batch_sequences = []

#         for i in range(num_parts):
#             curr_seq = sequences[:, i]
#             features, logits = cnn_model(curr_seq.view(B * S, C, H, W).to(device))
#             features = features.view(B, S, -1)
#             logits = logits.view(B, S, -1)
#             batch_features = torch.cat([features, logits], dim=-1)
#             batch_sequences.append(batch_features)

#         full_seq = torch.cat(batch_sequences, dim=-1)  # [1, 5, 6153]
#         print("Full Sequence Shape:", full_seq.shape)  # [1, 5, 6153]
#         lstm_features, logits, probs = lstm_model(full_seq)  # [1, 5, 512]
#         print("LSTM Features Shape:", lstm_features.shape)  # [1, 5, 512]
#         print("Logits:", logits)  # [1, 5, 1]

#         tsne = TSNE(n_components=2, perplexity=3, random_state=42)
#         # print("LSTM Features Shape:", lstm_features)  # [1, 5, 512]
#         lstm_tsne = tsne.fit_transform(lstm_features.squeeze(0).detach().cpu().numpy())  # → [5, 2]

#         # Plot
#         plt.figure(figsize=(6, 4))
#         plt.plot(lstm_tsne[:, 0], lstm_tsne[:, 1], marker='o')
#         for i, (x, y) in enumerate(lstm_tsne):
#             plt.text(x, y, str(i), fontsize=12)
#         plt.title("t-SNE of LSTM Features (Sequence of 5)")
#         plt.xlabel("Dim 1")
#         plt.ylabel("Dim 2")
#         plt.grid(True)
#         plt.tight_layout()

#         print("Shape", lstm_features.shape)

#         from sklearn.metrics.pairwise import cosine_similarity

#         # embs = lstm_features.squeeze(0).cpu().numpy()  # [5, 512]
#         # cos_sim_matrix = cosine_similarity(embs)       # [5, 5]

#         final_prob = probs.squeeze(0)[-1]
#         pred = (final_prob >= 0.5).int().item()
#         truth = label.item()

#         print("Pred:", pred, "Truth:", truth)
#         # print("Cosine Similarity Matrix:")
#         # print(np.round(cos_sim_matrix, 3))

#         all_embeddings.append(lstm_features[:, -1, :].squeeze(0).cpu().numpy())
#         all_labels.append(truth)
#         all_preds.append(pred)

#         # Color logic
#         if pred == truth and truth == 0:
#             color_map.append("blue")    # Correct 0
#         elif pred == truth and truth == 1:
#             color_map.append("red")     # Correct 1
#         elif pred == 1 and truth == 0:
#             color_map.append("green")   # False Positive
#         elif pred == 0 and truth == 1:
#             color_map.append("orange")  # False Negative

#         plt.show()

# # # === Global t-SNE over LSTM embeddings ===
# X_seq = np.stack(all_embeddings)

# tsne_seq = TSNE(n_components=2, random_state=42).fit_transform(X_seq)

# # === Plotting with custom colors and sequence # ===
# def plot_embedding_with_colors(embeddings, colors, title):
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, alpha=0.7, s=50)
#     plt.title(title)
#     plt.xlabel("Dim 1")
#     plt.ylabel("Dim 2")
#     plt.grid(True)

#     for idx, (x, y) in enumerate(embeddings):
#         plt.text(x + 0.2, y, str(idx), fontsize=8, ha='center', va='center', color='black')

#     # Custom legend
#     import matplotlib.patches as mpatches
#     legend_elements = [
#         mpatches.Patch(color='blue', label='Correct 0'),
#         mpatches.Patch(color='red', label='Correct 1'),
#         mpatches.Patch(color='green', label='Wrong: Pred 1, True 0'),
#         mpatches.Patch(color='orange', label='Wrong: Pred 0, True 1')
#     ]
#     plt.legend(handles=legend_elements)
#     plt.tight_layout()
#     plt.show()

# plot_embedding_with_colors(tsne_seq, color_map, "t-SNE of LSTM Embeddings by Prediction Outcome")

# All sequences (as 5 timesteps) Analsysis + Graphing
# -------------------------------------------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import torch

# all_embeddings = []
# all_seq_lengths = []
# all_labels = []
# all_preds = []
# color_map = []
# video_names = []

# with torch.no_grad():
#     for idx, (sequences, paths, label, info) in enumerate(dataloader_sequences):
#         B, num_parts, S, C, H, W = sequences.shape
#         batch_sequences = []

#         for i in range(num_parts):
#             curr_seq = sequences[:, i]
#             features, logits = cnn_model(curr_seq.view(B * S, C, H, W).to(device))
#             features = features.view(B, S, -1)
#             logits = logits.view(B, S, -1)
#             batch_features = torch.cat([features, logits], dim=-1)
#             batch_sequences.append(batch_features)

#         full_seq = torch.cat(batch_sequences, dim=-1)  # [B, S, feature_dim]
#         lstm_features, logits, probs = lstm_model(full_seq)  # [B, S, 512]

#         all_embeddings.append(lstm_features.cpu().numpy().reshape(-1, lstm_features.size(-1)))
#         all_seq_lengths.append(S)

#         for b in range(B):
#             final_prob = probs[b, -1]
#             pred = int((final_prob >= 0.5).item())
#             true_label = int(label[b].item())
#             all_labels.append(true_label)
#             all_preds.append(pred)

#             # Extract video name
#             first_path = paths[b][0][0]
#             video_name = first_path.split("/")[-1].split("_frame")[0]
#             video_names.append(video_name)

#             if pred == true_label == 0:
#                 color_map.append("blue")    # True Negative
#             elif pred == true_label == 1:
#                 color_map.append("red")     # True Positive
#             elif pred == 1 and true_label == 0:
#                 color_map.append("green")   # False Positive
#             elif pred == 0 and true_label == 1:
#                 color_map.append("orange")  # False Negative

# # Stack all embeddings into shape [total_timesteps, feature_dim]
# X_seq = np.vstack(all_embeddings)

# # Run t-SNE on all timesteps of all sequences
# tsne_seq = TSNE(n_components=2, random_state=42).fit_transform(X_seq)

# # Create line and marker style map based on unique video names
# unique_videos = sorted(set(video_names))
# line_styles = ['-', '--', '-.', ':']
# markers = ['o', 's', '^', 'D', '*', 'P', 'X', 'v', '<', '>']
# style_map = {
#     v: (line_styles[i % len(line_styles)], markers[i % len(markers)])
#     for i, v in enumerate(unique_videos)
# }

# # Plot
# plt.figure(figsize=(12, 10))
# start_idx = 0
# for i, seq_len in enumerate(all_seq_lengths):
#     seq_points = tsne_seq[start_idx:start_idx + seq_len]
#     video_name = video_names[i]
#     line_style, marker = style_map[video_name]

#     # Plot sequence line
#     plt.plot(seq_points[:, 0], seq_points[:, 1],
#              linestyle=line_style, color=color_map[i])

#     # Annotate sequence number slightly behind 0th point
#     direction = seq_points[-1] - seq_points[0]
#     norm = np.linalg.norm(direction)
#     unit_dir = direction / norm if norm != 0 else np.array([1, 0])
#     offset = unit_dir * 0.8
#     text_position = seq_points[0] - offset

#     plt.text(text_position[0], text_position[1], str(i),
#              fontsize=9, ha='center', va='center', color=color_map[i])

#     start_idx += seq_len

# plt.title("t-SNE of All LSTM Timesteps for All Sequences (Colored by TP/FP/TN/FN)")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# # -----------------------------------------------------------------------------------

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from tqdm import tqdm

# seen_images = set()
# cnn_embeddings = []
# frame_indices = []
# color_map = []

# class_colors = {
#     0: "blue",    # Class 0
#     1: "red",     # Class 1
#     2: "green"    # Class 2
# }

# frame_counter = 0

# with torch.no_grad():
#     for idx, (sequences, paths, label, info) in enumerate(tqdm(dataloader_sequences)):
#         B, P, S, C, H, W = sequences.shape

#         seen_paths = set()
#         for seq in paths[0]:  # paths[0] gives the full sequence
#             # print("Sequence paths:", seq)
#             for path_tuple in seq:
#                 # print("Path tuple:", path_tuple, type(p))
#                 img_path = path_tuple  # extract string from tuple
#                 # print("Image path:", img_path)
#                 if img_path not in seen_paths:
#                     seen_paths.add(img_path)
#                     # print("Unique image path:", img_path)

#                     img = Image.open(img_path).convert('RGB')  # Convert in case image is grayscale
#                     img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to device
#                     # You can now extract features, run t-SNE logic, store for plotting, etc.
#                     # print(img.shape)
#                     features, logits = cnn_model(img_tensor)
#                     probs = torch.softmax(logits, dim=1)
#                     pred_class = torch.argmax(probs, dim=1).item()

#                     cnn_embeddings.append(features.squeeze(0).cpu().numpy())
#                     frame_indices.append(frame_counter)
#                     color_map.append(class_colors[pred_class])

#                     frame_counter += 1

# # Run t-SNE on all CNN embeddings
# X = np.stack(cnn_embeddings)  # [num_frames, feature_dim]
# tsne_result = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# # Plot
# plt.figure(figsize=(12, 10))
# plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=color_map, s=20, alpha=0.7)

# # Annotate each point with frame #
# for i, (x, y) in enumerate(tsne_result):
#     plt.text(x + 0.3, y, str(frame_indices[i]), fontsize=6)

# # Legend
# import matplotlib.patches as mpatches
# legend_elements = [
#     mpatches.Patch(color='blue', label='Class 0'),
#     mpatches.Patch(color='red', label='Class 1'),
#     mpatches.Patch(color='green', label='Class 2')
# ]
# plt.legend(handles=legend_elements, loc='right')

# plt.title("t-SNE of Individual CNN Frame Embeddings")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ----------------------------------------------

transition_windows = {
    "08_22_2022_12_54_56_fullvid.MP4": [(78453, 563, 732), (78950, 688, 844), (78991, 883, 844), (79021, 634, 799)],
    "10_31_2023_14_36_44_fullvideo.mp4": [(4585, 695, 921), (5241, 826, 981), (7432, 690, 555), (8230, 845, 619), (9950, 741, 512),  
                                          (10881, 817, 952), (12600, 982, 787), (13725, 667, 968), (14870, 812, 613)],
}

transition_windows_fold1 = {
    "01_30_2023_07_11_48_fullvideo.MP4": [(58365, 510, 695)],
    "02_02_2024_08_41_58_fullvid.mp4": [(20982, 972, 602), (21098, 745, 614), (21229, 956, 729), (22026, 748, 666), (22327, 858, 780), 
                                        (24008, 910, 729), (24873, 530, 937)]
}

transition_windows_fold2 = {
    "02_05_2021_16_32_47_fullvideo.mp4": [(7894, 927, 968), (8635, 583, 680)],
    "08_22_2022_12_54_56_fullvid.MP4": [(78453, 938, 840), (78950, 542, 606), (78991, 681, 944), (79021, 566, 786)],
}

transition_windows_fold3 = {
    "09_30_2022_11_38_00_fullvid.MP4": [(742, 971, 665), (52674, 859, 562), (53048, 676, 513), (53082, 859, 721), 
                                        (53119, 930, 989), (53728, 856, 650)],
    "10_31_2022_07_18_14_fullvid.MP4": [(505, 939, 835), (526, 918, 695), (640, 873, 506), (61887, 598, 842)],
}

transition_windows_fold4 = {
    "10_31_2023_14_36_44_fullvideo.mp4": [(4585, 875, 531), (5241, 627, 938), (7432, 882, 823), (8230, 775, 775), (9950, 642, 753), 
                                            (10881, 687, 568), (12600, 827, 849), (13725, 706, 617), (14870, 795, 615)],
    "12_22_2022_06_48_13_fullvid.MP4": [(18846, 531, 862), (19463, 915, 950), (34249, 686, 662), (37824, 666, 553), (38806, 586, 626), 
                                        (40875, 575, 662), (44452, 858, 806), (52698, 545, 596), (53242, 907, 548), (55451, 964, 500), 
                                        (59200, 777, 657), (59488, 798, 686)],
}

transition_windows_fold5 = {
    "01_30_2023_07_11_48_fullvideo.MP4": [(58365, 505, 649)],
    "08_22_2022_12_54_56_fullvid.MP4": [(78453, 880, 500), (78950, 701, 748), (78991, 676, 560), (79021, 967, 675)],
}

transition_windows_fold6 = {
    "01_30_2023_07_11_48_fullvideo.MP4": [(58365, 710, 510)],
    "09_30_2022_11_38_00_fullvid.MP4": [(742, 949, 951), (52674, 891, 735), (53048, 613, 696), (53082, 512, 664), 
                                        (53119, 669, 755), (53728, 723, 863)],
}

image_paths = [
    os.path.join(DATA_PATH, f)
    for f in os.listdir(DATA_PATH)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]

filtered_paths = get_relevant_frames_per_transition(image_paths, transition_windows)
dataset = OrderedFrameDataset(filtered_paths, transform=transform)
print(dataset.__len__(), "filtered paths")
dataloader_images = DataLoader(dataset, batch_size=1, shuffle=False)

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
import torch

# === Utilities ===

def extract_frame_number(path):
    match = re.search(r'frame_(\d+)\.jpg', path)
    return int(match.group(1)) if match else -1

def extract_video_id(fname):
    return fname.split('_frame_')[0]  # everything before frame_ = video ID

# === Main ===

frame_counter = 0
cnn_embeddings = []
frame_indices = []
filenames = []
color_map = []
predicted_classes = []
seen_paths = set()

# Define class color mapping
class_colors = {
    0: 'blue',
    1: 'red',
    2: 'green'
}

predicted_transition_frames = []
predicted_classes = []
filenames = []
frame_numbers = []
video_ids = []

# Keep track of last 5 predictions
from collections import deque
# num_steps = 6  # Number of steps to consider for stability
num_steps = 12
prev_preds = deque(maxlen=num_steps)
prev_frame_nums = deque(maxlen=num_steps)
prev_video_ids = deque(maxlen=num_steps)
stable_class = None

with torch.no_grad():
    print("Dataloader length:", len(dataloader_images))
    for frame, path in dataloader_images:
        img_tensor = frame.to(device)
        img_path = path[0]
        base_name = os.path.basename(img_path)

        if img_path not in seen_paths:
            seen_paths.add(img_path)

            features, logits = cnn_model(img_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

            frame_num = extract_frame_number(img_path)
            video_id = extract_video_id(img_path)

            # Append to tracking lists
            filenames.append(base_name)
            frame_numbers.append(frame_num)
            video_ids.append(video_id)
            predicted_classes.append(pred_class)
            cnn_embeddings.append(features.squeeze(0).cpu().numpy())
            color_map.append(class_colors[pred_class]) # type: ignore

            # Update history
            prev_preds.append(pred_class)
            prev_frame_nums.append(frame_num)
            prev_video_ids.append(video_id)

            # Transition check: only if we have full history
            if len(prev_preds) == num_steps:
                if stable_class is None:
                    stable_class = prev_preds[-1]

                elif pred_class != stable_class:
                    # Check if last 5 predictions are all new class
                    if all(p == pred_class for p in prev_preds):
                        # Check if frame numbers are sequential
                        is_sequential = all(prev_frame_nums[i] + 1 == prev_frame_nums[i + 1] for i in range(num_steps - 1))

                        # Check if same video
                        is_same_video = len(set(prev_video_ids)) == 1

                        if is_sequential and is_same_video:
                            predicted_transition_frames.append(len(predicted_classes) - 1)
                            stable_class = pred_class  # Update stable class

            print(f"Processed {base_name} - Frame {frame_num}, Predicted Class: {pred_class}")

# === Run t-SNE ===
X = np.stack(cnn_embeddings)
tsne_result = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

# === Plotting ===
plt.figure(figsize=(12, 10))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=color_map, s=20, alpha=0.7)

for i, (x, y) in enumerate(tsne_result):
    label = filenames[i]
    if i in transition_frames:
        label += " ★"
    plt.text(x + 0.3, y, label, fontsize=6)

# Legend
legend_elements = [
    mpatches.Patch(color='blue', label='Class 0'),
    mpatches.Patch(color='red', label='Class 1'),
    mpatches.Patch(color='green', label='Class 2')
]
# plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5))
# plt.title("t-SNE of CNN Frame Embeddings with Detected Transitions")
# plt.xlabel("Dim 1")
# plt.ylabel("Dim 2")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# === Print Transition Frames ===
print("\nTransition Frame Filenames (stable class switch + sequential + same video):")
for i in predicted_transition_frames:
    print(f"{i} → {filenames[i]}")