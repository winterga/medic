import torch
import torch.nn as nn
from torchvision.transforms import v2
import torch.utils.data as data
import torch.optim as optim
import time, os, copy

import tqdm
import random
import sys
import re

from statistics import mean, stdev
from .truth_data import transition_frames

from .dataset_types import AlternatingSequenceDataset, SequentialSequenceDataset
from .model_types import CNNModel, TraditionalBidirectionalLSTM
from .SSP import SSP

import torch.nn.functional as F
from pathlib import Path

from sklearn.metrics import f1_score

def load_datasets(params, hyper_params):
    cpu_train_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    cpu_valid_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    cpu_test_list = [
        v2.Resize(size=(hyper_params['img_size'], hyper_params['img_size'])),
        v2.ToTensor(),
    ]
    
    if hyper_params['normalize']:
        cpu_train_list.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        cpu_valid_list.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        cpu_test_list.append(v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))

    image_transforms = {
        'train': v2.Compose(cpu_train_list),
        'valid': v2.Compose(cpu_valid_list),
        'test': v2.Compose(cpu_test_list)
    }
    
    # Load data from folders
    dataset_map = {
        'alternating': AlternatingSequenceDataset,
        'sequential': SequentialSequenceDataset
    }
    try:
        dataset_class = dataset_map[hyper_params['training_style']]
    except KeyError:
        raise ValueError(f"Invalid training_style: {hyper_params['training_style']}")

    train_dataset = dataset_class(
        root_dir=params['train_dir'],
        transform=image_transforms['train'],
        transition_frames=transition_frames
    )
    dataset = {
        'train': train_dataset,
        'valid': SequentialSequenceDataset(root_dir=params['valid_dir'], transform=image_transforms['valid'], transition_frames=transition_frames),
        'test': SequentialSequenceDataset(root_dir=params['test_dir'], transform=image_transforms['test'], transition_frames=transition_frames)
    }

    dataset['valid'].restrict_to_transition_windows(min_context=500, max_context=1000)
    dataset['test'].restrict_to_transition_windows(min_context=500, max_context=1000)

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'valid': data.DataLoader(dataset['valid'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'test': data.DataLoader(dataset['test'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True)
    }
    return dataloaders

def load_train(params, hyper_params):
    dataloaders = load_datasets(params, hyper_params)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Define CNN
    cnn_model = CNNModel()
    cnn_feature_dim = cnn_model.get_feature_dim()
    cnn_class_dim = cnn_model.get_num_classees()

    # 2. Define SSP
    sample_seq = dataloaders['train'].dataset[0][0] # [S, C, H, W]
    seq_length = sample_seq.shape[0]
    ssp = SSP(sequence_length=seq_length, window_size=1, frame_gap=1, stride=1)
    num_temporal_timesteps = ssp.num_temporal_timesteps

    # 3. Define input_dim
    ## `cnn_dim` * `ssp.window_size` plus adding the logits per class for more information
    add_dim = cnn_class_dim * ssp.window_size
    input_dim = cnn_feature_dim * ssp.window_size + add_dim
    # Using hyper_params, it's easier to store a false. So, we subtract if it is true instead of having to make each execution be `--add_logits`
    if hyper_params['no_add_logits']:
        input_dim -= add_dim
    print(f"Input Dim: {input_dim}")

    # 4. Define the LSTM model
    model_ft = TraditionalBidirectionalLSTM(input_dim).to(device)
    # class_weights = torch.tensor([1, 10]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay']) # type: ignore
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model_ft, cnn_model, dataloaders, criterion, optimizer, scheduler, device, ssp


def train_model(params, hyper_params):
    since = time.time()

    model, cnn_model, dataloaders, criterion, optimizer, scheduler, device, ssp = load_train(params, hyper_params)
    model.to(device)
    print(f"Train: {next(model.parameters()).device}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_epoch = 0
    best_f1 = -1.0

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid', 'test']:

            (model.train if phase == 'train' else model.eval)()

            if phase == 'test' and not best_epoch == epoch:
                print(f"Skipping test phase for epoch {epoch} as it is not the best epoch ({best_epoch})")
                continue

            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}', unit='batch') as pbar:
                mod = 5
                i = 0

                epoch_loss = 0
                epoch_loss_0s = 0
                epoch_loss_1s = 0
                total_correct_sequences = 0
                total_correct_sequences_0s = 0
                total_correct_sequences_1s = 0
                total_sample_sequences = 0
                total_sample_sequences_0s = 0
                total_sample_sequences_1s = 0
                predicted_sequences_count = 0
                truth_sequence_count = 0

                # Sequences to transitions calculation
                """
                1. We calculate a transition as a set of 2-3 sequential sequences. Why 2-3?:
                Permitted frames to be considered a sequence: [4], [0-4], [0]
                - Batches of 3 sequences are moved left 3 frames
                - That means if the image was right-[0], it would then be mid-[2], and then be [4]. 
                
                2. We can't count directly, because it could be a set of sequences where [1, 0, 1] and this could be a sequence.
                But, we also don't want to hardcode the value and say [1, 0, 1, 1, 1] as 2 sequences. 
                a. See a [1]
                    a. See [1] before seeing 2 [0]s
                        a. Transition ends at seeing 2 consecutive [0]s
                    b. See 2 [0]s before seeing another [1]
                        a. No transition
                b. See a [0]
                    a. No transition
                **THIS STATES THAT [1, 1, 1, 1, 0, 1, 1, 1] WOULD BE ONE (1) SEQUENCE**

                3. We need to then look at the paths that annotate a transition to consider an EXACT frame as the transition point

                """
                current_sequence_labels = []
                transition_count = 0
                correct_transition_count = 0

                video_transitions_frames = {}

                ## PLOTTING
                all_features = []
                all_labels = []
                all_preds = []
                ## PLOTTING

                # Iterate over data.
                for j, (sequences, paths, label, info) in enumerate(dataloaders[phase]):

                    # Train should be all transition sequences, but validation/test should be every 5th
                    if phase == 'train' or i % mod == 0:
                        sequences = sequences.to(device, non_blocking=True) 
                        label = label.to(device, non_blocking=True).float()
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            # flatten across parts so we always have [B, SL, C, H, W]
                            B, S, C, H, W = sequences.shape  # Batch size, sequence length, channels, height, width

                            batch_preds = []
                            batch_sequence = []

                            # Iterate over SSP windows
                            print(f"Windows Indices: {ssp.get_windows()}")
                            for window_indices in ssp.get_windows():  # SSP should compute windows based on sequence length
                                seq_window = sequences[:, window_indices]  # [B, W, C, H, W], W = window size
                                
                                # Feed all frames in window into CNN
                                features, logits = cnn_model(seq_window.view(B * len(window_indices), C, H, W))  # [B*W, F], [B*W, 3]
                                features = features.view(B, len(window_indices), -1)  # [B, W, F]
                                logits = logits.view(B, len(window_indices), -1)      # [B, W, 3]

                                # Optional: store class predictions for analysis
                                probs = F.softmax(logits, dim=-1)                     # [B, W, 3]
                                _, preds = torch.max(probs, dim=-1)                  # [B, W]
                                batch_preds.append(preds.tolist())

                                # Concatenate features and logits if desired
                                combined = features if hyper_params['no_add_logits'] else torch.cat([features, logits], dim=-1)  # [B, W, F(+3)]
                                batch_sequence.append(combined)

                            # Concatenate all SSP windows along the time dimension
                            full_sequence = torch.cat(batch_sequence, dim=1)  # [B, total_timesteps, F(+3)]
                            print(f"Full SSP sequence shape: {full_sequence.shape}")

                            # Feed into your LSTM / SSP temporal model
                            lstm_features, outputs, probs = model(full_sequence)  # [B, 1] or [B, T, 1] depending on model
                            all_features.append(lstm_features.cpu().detach().numpy())
                            all_labels.append(label.cpu().detach().numpy())
                            loss = criterion(outputs, label)  # [B, 1] vs [B, 1]
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            
                            epoch_loss += loss.item()
                            epoch_loss_0s += loss.item() if label.item() == 0 else 0
                            epoch_loss_1s += loss.item() if label.item() == 1 else 0
                            
                            preds = (outputs > 0.5).float()  # Convert logits to binary predictions (0 or 1)
                            all_preds.append(preds.cpu().detach().numpy())  # Store predictions for later analysis
                            file_names = [os.path.basename(p[0]) for p in paths]
                            print(f"Info: {outputs.item():.6f}", preds.item(), label.item(), batch_preds, info)
                            print(f"F: {file_names}")
                            
                            total_correct_sequences += torch.sum(preds == label.data)  # Count correct predictions
                            total_correct_sequences_0s += torch.sum((preds == label.data) & (label.data == 0))
                            total_correct_sequences_1s += torch.sum((preds == label.data) & (label.data == 1))
                            total_sample_sequences += label.size(0)  # Increment total samples
                            total_sample_sequences_0s += (label == 0).sum().item()
                            total_sample_sequences_1s += (label == 1).sum().item()
                            predicted_sequences_count += (preds.item() == 1)
                            truth_sequence_count += (label.item() == 1)
                            correct_transition_count += (preds.item() == label.item() == 1)

                            # Always start with 1 or add (0 | 1) to a current sequence
                            if phase != 'train':
                                if not (preds.item() == 0 and len(current_sequence_labels) == 0):
                                    current_sequence_labels.append((preds.item(), paths))
                                    curr_preds = [p[0] for p in current_sequence_labels]
                                    paths = [p[1] for p in current_sequence_labels][0]


                                    # curr_preds is a list of predictions for the current sequence
                                    # Check if the final two (2) of curr_preds `curr_preds[-2:]` ends in [0, 0] -- end of transition
                                    if curr_preds[-2:] == [0, 0] or j == int(len(dataloaders[phase]) / mod) * mod:

                                        # We need to be sure to account for the modular -- for example, if mod == 5, then we need to check if the last batch is the last batch
                                        # We only need to count the ones where it is > 2 (or do we? - consider that an image should be in 1/2/3 transitions?)
                                        if curr_preds.count(1) >= 2 or j == int(len(dataloaders[phase]) / mod) * mod:
                                            # Count transitions
                                            transition_count += 1
                                        

                                            # Print total transition frame start and end
                                            path_start = paths[0][0][0]
                                            path_end = paths[2][4][0]
                                            print(f"Transition located starting @ path_start: {path_start} and end: {path_end}")

                                            # Predict the frame using the middle of the sequence.
                                            """
                                            EXAMPLE: path_start: ['/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/10_31_2022_07_18_14_fullvid.MP4_frame_00496.jpg'] and end: ['/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/10_31_2022_07_18_14_fullvid.MP4_frame_00510.jpg']
                                            """

                                            ## First frame  
                                            first_pathname = Path(path_start).name  # Get just the filename
                                            first_match = re.match(r"(.*\.MP4)_frame_(\d+)\.jpg", first_pathname, re.IGNORECASE)
                                            second_pathname = Path(path_end).name  # Get just the filename
                                            second_match = re.match(r"(.*\.MP4)_frame_(\d+)\.jpg", second_pathname, re.IGNORECASE)

                                            if first_match and second_match:
                                                first_video_name = first_match.group(1)
                                                first_frame_number = int(first_match.group(2))
                                                second_video_name = second_match.group(1)
                                                second_frame_number = int(second_match.group(2))
                                                if first_video_name != second_video_name:
                                                    print("Warning: Video names do not match!")
                                                    sys.exit(1)
                                                if first_frame_number >= second_frame_number:
                                                    print("Warning: First frame number is not less than second frame number!")
                                                    sys.exit(1)

                                                # Calculate the middle frame number
                                                middle_frame_number = (first_frame_number + second_frame_number) // 2
                                                video_transitions_frames.setdefault(first_video_name, []).append(middle_frame_number)

                                            else:
                                                print("No match found")

                                            # Associate the frame to the specific video and then count

                                        # Clear the sequences
                                        current_sequence_labels = []

                    i += 1                    
                    pbar.update(1)

                # Best Valid Epoch to run test against
                avg_loss = ((epoch_loss_0s/total_sample_sequences_0s) + (epoch_loss_1s/total_sample_sequences_1s)) / 2
                print(len(all_labels), len(all_preds))
                macro_f1 = f1_score(all_labels, all_preds, average='macro')
                # if phase == 'valid' and (avg_loss) < best_loss:
                if phase == 'valid' and macro_f1 > best_f1:
                    best_loss = avg_loss
                    best_f1 = macro_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(params['save_dir'], f'{params["name"]}.pth'))
                    print(f"Best model weights updated @ Epoch {epoch} and saved to {params['save_dir']}/{params['name']}.pth")

                print(f"Avg Epoch {phase} {epoch} Loss", epoch_loss / total_sample_sequences)
                print(f"Avg Epoch {phase} {epoch} 0s Loss", epoch_loss_0s / total_sample_sequences_0s)
                print(f"Avg Epoch {phase} {epoch} 1s Loss", epoch_loss_1s / total_sample_sequences_1s)
                print(f"Avg Epoch {phase} {epoch} Loss (0s + 1s)/2", avg_loss)
                print(f"Macro F1 {phase} {epoch} Score: {macro_f1:.4f}")
                accuracy = total_correct_sequences / total_sample_sequences
                print(f"{phase.capitalize()} Accuracy: {accuracy * 100:.2f}%")
                print(f"{phase.capitalize()} Total Correct Sequences: {total_correct_sequences} out of {total_sample_sequences}")
                print(f"{phase.capitalize()} Total Correct 0s: {total_correct_sequences_0s} out of {total_sample_sequences_0s}")
                print(f"{phase.capitalize()} Total Correct 1s: {total_correct_sequences_1s} out of {total_sample_sequences_1s}")
                print(f"***SEQUENCE DATA EPOCH {epoch} {phase}***")
                print("# of Predicted Sequences:\t", predicted_sequences_count)
                print("# of Truth Sequences:\t\t", truth_sequence_count)
                # [p:a] - [0:1] does not work; only [1:1]
                print("Predicted Correct Transitions: ", correct_transition_count)
                print(f"***SEQUENCE DATA EPOCH {epoch} {phase}***")

                # Note about statistics
                """
                predicted_sequences_count: Number of sequences that were predicted as transitions (1)
                truth_sequence_count: Number of sequences that were actually transitions (1)
                correct_transition_count: Number of sequences that were correctly predicted as transitions (1)
                """

                print('')
                
                print(f"***FRAME DATA EPOCH {epoch} {phase}***")
                print(dataloaders[phase].dataset.transition_frames) # type: ignore
                total_unmatched_truths = 0
                total_unmatched_preds = 0
                all_matched_pairs = []
                for video_name, truth_frames in dataloaders[phase].dataset.transition_frames.items(): # type: ignore
                    truths = truth_frames
                    preds = video_transitions_frames.get(video_name, [])
                    matched_pairs, unmatched_truths, unmatched_preds = match_predictions_to_truths(truths, preds)
                    print('Video: ', video_name)
                    print('Truth Frames: \t', truths)
                    print('Pred Frames: \t', preds)
                    print('Match Frames: \t', matched_pairs)
                    print('Unm Truths: ', unmatched_truths)
                    print('Unm Preds:', unmatched_preds)

                    total_unmatched_truths += len(unmatched_truths)
                    total_unmatched_preds += len(unmatched_preds)
                    all_matched_pairs.extend(matched_pairs)
                # print("# of Predicted Frames: ", transition_count)
                # print("# of Truth Frames: ", dataloaders[phase].dataset.transition_frames)
                # print("Video Transitions Frames: ", video_transitions_frames)
                print('# of Unmatched Truth Frames: ', total_unmatched_truths)
                print('# of Unmatched Pred Frames: ', total_unmatched_preds)
                distances = [abs(t - p) for t, p in all_matched_pairs]

                # Compute statistics
                if len(distances) > 0:
                    avg_distance = mean(distances)
                    min_distance = min(distances)
                    max_distance = max(distances)

                    print(f"Average distance: {avg_distance}")
                    print(f"Minimum distance: {min_distance}")
                    print(f"Maximum distance: {max_distance}")
                if len(distances) > 1:
                    std_dev_distance = stdev(distances)
                    print(f"Standard deviation: {std_dev_distance}")
                else: 
                    print("Standard deviation: Not enough data to compute")
                print(f"***FRAME DATA EPOCH {epoch} {phase}***")

            scheduler.step()

            print(f"PHASE {phase} DONE")
            print()

        print()
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val loss: {best_loss:.4f} at epoch {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    # experiment.end()
    return model

# Leniency of 30 frames (or 1 second) is given to best match
def match_predictions_to_truths(truths, preds, max_distance=30):
    truths = sorted(truths)
    preds = sorted(preds)
    matched_truths = set()
    matched_preds = set()
    matched_pairs = []

    for pred in preds:
        best_match = None
        best_distance = float('inf')
        for i, truth in enumerate(truths):
            if i in matched_truths:
                continue  # already matched
            distance = abs(pred - truth)
            if distance <= max_distance and distance < best_distance:
                best_match = i
                best_distance = distance
        if best_match is not None:
            matched_truths.add(best_match)
            matched_preds.add(pred)
            matched_pairs.append((truths[best_match], pred))

    unmatched_truths = [truths[i] for i in range(len(truths)) if i not in matched_truths]
    unmatched_preds = [pred for pred in preds if pred not in matched_preds]

    return matched_pairs, unmatched_truths, unmatched_preds