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

import torch.nn.functional as F
from pathlib import Path

from sklearn.metrics import f1_score

def load_train(params, hyper_params):
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

    # GPU-based transforms for heavy augmentations
    gpu_train_list = []
    
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

    # Size of train and validation data
    dataset_sizes = {
        'train': len(dataset['train']),
        'valid': len(dataset['valid']),
        'test': len(dataset['test'])
    }

    # Create iterators for data loading
    dataloaders = {
        'train': data.DataLoader(dataset['train'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'valid': data.DataLoader(dataset['valid'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True),
        'test': data.DataLoader(dataset['test'], batch_size=hyper_params['batch_size'], shuffle=False,
                            num_workers=hyper_params['cpu_count'], pin_memory=True, drop_last=True)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cnn_model = CNNModel()
     # The model needs to accept 4 different sizes:
    # 1. 3x5 (5 images, 3 parts) w/ Logits          -> 3  * (2048 + 3)  = 6153
    # 2. 15 (15 images, single part) w/ Logits      -> 15 * (2048 + 3)  = 30765
    # 3. 3x5 (5 images, 3 parts) w/out Logits       -> 3  * (2048)      = 6144
    # 4. 15 (15 images, single part) w/out Logits   -> 15 * (2048)      = 30720
    input_dim = 0
    print(hyper_params['images_in_batch'])
    if hyper_params['images_in_batch'] == '3x5' and hyper_params['no_add_logits'] :
        input_dim = 6144
    elif hyper_params['images_in_batch'] == '3x5' and not hyper_params['no_add_logits']:
        input_dim = 6153
    elif hyper_params['images_in_batch'] == '15' and hyper_params['no_add_logits']:
        input_dim = 2048
    elif hyper_params['images_in_batch'] == '15' and not hyper_params['no_add_logits']:
        input_dim = 2051
    model_ft = TraditionalBidirectionalLSTM(input_dim).to(device)
    activation = nn.Softmax(dim=1)
    # class_weights = torch.tensor([1, 10]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.5]).to(device))
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'], weight_decay=hyper_params['weight_decay']) # type: ignore
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model_ft, cnn_model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list


def train_model(params, hyper_params):
    since = time.time()

    model, cnn_model, dataloaders, data_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list = load_train(params, hyper_params)
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
            # criterion = nn.CrossEntropyLoss(weight=train_weights if phase == 'train' else valid_weights)
            # model.train() if phase == 'train' else model.eval()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   

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

                            # Example input to CNN
                            # sequences: [Batch, 3, Sequence, Channel, H, W]
                            B, num_parts, S, C, H, W = sequences.shape  # Batch size, sequence length, channels, height, width
                            print("Shape of sequences:", sequences.shape)

                            batch_preds = []
                            batch_sequence = []
                            full_sequence = None

                            if(hyper_params['images_in_batch'] == '3x5'):
                                # batch_paths = path
                                for i in range(num_parts):  # prev, curr, next
                                    seq = sequences[:, i]  # [B, S, C, H, W]
                                    features, logits = cnn_model(seq.view(B * S, C, H, W))  # [B*S, F], [B*S, 3]
                                    features = features.view(B, S, -1)  # [B, S, F]
                                    logits = logits.view(B, S, -1)  # [B, S, 3]
                                    probs = F.softmax(logits, dim=-1)  # [B, S, 3] - softmax across the class dimension

                                    # Get the predicted class index` (0, 1, or 2)
                                    _, preds = torch.max(probs, 2)
                                    batch_preds.append(preds.tolist())  # [B, S, 3]

                                    # Concatenate features and predictions
                                    if hyper_params['no_add_logits']:
                                        batch_sequence.append(features)  # [B, S, F]
                                    else:
                                        combined = torch.cat([features, logits], dim=-1)  # [B, S, F+3]
                                        batch_sequence.append(combined)

                                full_sequence = torch.cat(batch_sequence, dim=-1)  # [B, S, 3*(F+3)]
                            
                            elif (hyper_params['images_in_batch'] == '15'):
                                # sequences: [B, 3, S, C, H, W] → merge across 3 to form flat [B, 15, C, H, W]
                                flat_sequence = sequences.view(B, 3 * S, C, H, W)  # [B, 15, C, H, W]
                                # print("SHAPE", flat_sequence.shape)
                                
                                # Flatten batch and sequence to feed into CNN
                                seq = flat_sequence  # [B, 15, C, H, W]
                                features, logits = cnn_model(seq.view(B * 15, C, H, W))  # [B*15, F], [B*15, 3]

                                # Reshape back to batch form
                                features = features.view(B, 15, -1)  # [B, 15, F]
                                logits = logits.view(B, 15, -1)      # [B, 15, 3]
                                probs = F.softmax(logits, dim=-1)    # [B, 15, 3]

                                _, preds = torch.max(probs, 2)       # [B, 15]
                                batch_preds.append(preds.tolist())

                                if hyper_params['no_add_logits']:
                                    full_sequence = features  # [B, 15, F]
                                else:
                                    full_sequence = torch.cat([features, logits], dim=-1)  # [B, 15, F+3]

                            else:
                                print("NONEXISTENT")
                                sys.exit(1)

                            lstm_features, outputs, probs = model(full_sequence)  # [B, 1]
                            all_features.append(lstm_features.cpu().detach().numpy())  # Store LSTM features for later analysis
                            all_labels.append(label.cpu().detach().numpy())  # Store labels for later analysis
                            loss = criterion(outputs, label)  # [B, 1] vs [B, 1]

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            
                            epoch_loss += loss.item()
                            epoch_loss_0s += loss.item() if label.item() == 0 else 0
                            epoch_loss_1s += loss.item() if label.item() == 1 else 0
                            
                            preds = (outputs > 0.5).float()  # Convert logits to binary predictions (0 or 1)
                            all_preds.append(preds.cpu().detach().numpy())  # Store predictions for later analysis
                            filenames = [[p[0].split('/')[-1] for p in group] for group in paths]
                            print(f"{outputs.item():.6f}", preds.item(), label.item(), batch_preds, info)
                            for group in filenames:
                                print([path for path in group])
                            
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

# 1. Run Experiment
# 2. Get Model from checkpoint
# 3. Place those 15 specific images into TSNE for the model.