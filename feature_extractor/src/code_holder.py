```
class SequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, transition_frames=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.transition_frames = transition_frames

        # Load and sort all image paths
        self.image_paths = [os.path.join(root_dir, fname) 
                            for fname in os.listdir(root_dir) 
                            if fname.endswith(('.png', '.jpg'))]
        self.image_paths = sorted(self.image_paths, key=self.natural_sort_key)

        # Group by video name
        self.video_to_frames = {}
        for path in self.image_paths:
            video_name = self.extract_video_name(path)
            if video_name not in self.video_to_frames:
                self.video_to_frames[video_name] = []
            self.video_to_frames[video_name].append(path)

        self.triplet_sequences = []
        for video, frames in self.video_to_frames.items():
            num_frames = len(frames)
            triplet_span = 3 * sequence_length
            for i in range(num_frames - triplet_span + 1):
                # Slice into prev, curr, next
                prev_seq = frames[i : i + sequence_length]
                curr_seq = frames[i + sequence_length : i + 2 * sequence_length]
                next_seq = frames[i + 2 * sequence_length : i + 3 * sequence_length]
                full_seq = prev_seq + curr_seq + next_seq

                if self.is_sequential(full_seq):
                    self.triplet_sequences.append((video, prev_seq, curr_seq, next_seq))

    def natural_sort_key(self, path):
        match = re.search(r'(?P<video_name>.*?)(?P<frame_number>\d+)(?:\.jpg|\.png)', os.path.basename(path))
        if match:
            return (match.group('video_name'), int(match.group('frame_number')))
        return (path, 0)

    def extract_video_name(self, path):
        match = re.search(r'(.*?)(_frame_)\d+', os.path.basename(path))
        return match.group(1) if match else None

    def extract_frame_number(self, path):
        match = re.search(r'_frame_(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else -1

    def is_sequential(self, frame_paths):
        frame_numbers = [self.extract_frame_number(p) for p in frame_paths]
        expected = list(range(frame_numbers[0], frame_numbers[0] + len(frame_numbers)))
        return frame_numbers == expected
    
    def get_transition_positions(self, video_name, seq, positions=None):
        """
        Return the positions (indices) within `seq` that match transition frames.
        If `positions` is provided, only those indices are checked (can include negatives).
        """
        if video_name not in self.transition_frames:
            return []

        transition_set = set(self.transition_frames[video_name])
        seq_len = len(seq)

        if positions is None:
            indices_to_check = range(seq_len)
        else:
            indices_to_check = [
                i if i >= 0 else seq_len + i
                for i in positions
                if -seq_len <= i < seq_len
            ]

        matching_positions = []
        for i in indices_to_check:
            frame_num = self.extract_frame_number(seq[i])
            if frame_num in transition_set:
                matching_positions.append(int(i))  # 👈 ensure it's a Python int

        return matching_positions
    
    def contains_transition(self, video_name, seq, positions=None):
        """
        Check if the sequence contains a transition frame.
        
        If `positions` is provided, only check those indices in the sequence.
        Otherwise, check the whole sequence.
        """
        if video_name not in self.transition_frames:
            return False

        transition_set = set(self.transition_frames[video_name])
        print("Transition Set:", transition_set)

        if positions is None:
            frame_numbers = [self.extract_frame_number(p) for p in seq]
        else:
            frame_numbers = [self.extract_frame_number(seq[i]) for i in positions if 0 <= i < len(seq)]

        return any(f in transition_set for f in frame_numbers)

    def __len__(self):
        return len(self.triplet_sequences)

    def __getitem__(self, idx):
        video_name, prev_seq, curr_seq, next_seq = self.triplet_sequences[idx]
        sequences = [prev_seq, curr_seq, next_seq]

        all_images = []
        all_paths = []

        for seq in sequences:
            images = []
            for path in seq:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            all_images.append(torch.stack(images))  # shape: (sequence_length, C, H, W)
            all_paths.append(seq)

        images_tensor = torch.stack(all_images)  # shape: (3, sequence_length, C, H, W)

        # Label = does the current sequence contain a transition?
        # Get transition positions
        prev_positions = self.get_transition_positions(video_name, prev_seq, positions=[-2, -1])
        curr_positions = self.get_transition_positions(video_name, curr_seq)  # check all
        next_positions = self.get_transition_positions(video_name, next_seq, positions=[0, 1])

        # Binary label
        transition_label = int(bool(prev_positions or curr_positions or next_positions))

        transition_info = {
            "prev": prev_positions,
            "curr": curr_positions,
            "next": next_positions
        }

        return images_tensor, all_paths, transition_label, transition_info

class StrongTransitionLSTM(nn.Module):
    def __init__(self, feature_dim=6150, num_classes=3, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        feature_dim: Dimension of CNN feature output (2048 for ResNet-50's second-to-last layer)
        num_classes: Number of classes predicted by the CNN (3 for your 3 classes: 0, 1, 2)
        """
        super().__init__()
        
        # Update input_dim to the combined size (features + predictions)
        input_dim = feature_dim + num_classes  # Combine CNN features and predictions
        
        # LSTM setup for sequence input (CNN features and predictions combined)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # Fully connected layer to output transition score
        self.fc = nn.Linear(hidden_dim * 2, 1)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)

        self.apply_xavier_init()

    def apply_xavier_init(self):
        # Apply Xavier initialization to LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)  # or init.xavier_normal_(param)
            elif 'bias' in name:
                init.zeros_(param)  # Initialize biases to zero

                hidden_size = param.shape[0] // 4
                param.data[hidden_size:2*hidden_size] = 1.0
        
        # Apply Xavier initialization to fully connected layer
        init.xavier_uniform_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x):
        """
        x: Input tensor of combined CNN features and predictions (shape: [B, S, input_dim])
        """
        # Forward pass through the LSTM layer
        out, _ = self.lstm(x)  # (B, seq_len, 2*hidden_dim)
        
        # Apply dropout to the last time step
        out = self.dropout(out[:, -1, :])  # Last timestep
        
        # Output from fully connected layer
        out = self.fc(out)
        
        # Sigmoid output for binary classification (transition/no-transition)
        return torch.sigmoid(out).squeeze(1)  # Output shape: (B,)
    
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.original_cnn_model = torch.load("/home/user/Documents/GitHub/medic/feature_extractor/checkpoints/Resnet50_022125_12/Resnet50_022125_12.pth")

        # Freeze all parameters in the original model
        for param in self.original_cnn_model.parameters():
            param.requires_grad = False

        # Remove the final classification layers (usually the last fully connected layer)
        self.features = nn.Sequential(*list(self.original_cnn_model.children())[:-1])
        self.predictions = self.original_cnn_model.fc

    def forward(self, x):
        # Extract features from the second-to-last layer
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        y = self.predictions(x)
        return x, y

def get_subset_indices(dataset, subset_percentage=0.1):
    """
    Get indices for a subset of the dataset (e.g., 10% of the data).
    """
    # Randomly shuffle indices
    indices = list(range(len(dataset)))
    subset_size = int(len(indices) * subset_percentage)
    subset_indices = random.sample(indices, subset_size)  # Randomly sample 10% of the data
    return subset_indices

def apply_gpu_transforms(image, transform_list):
    """
    Apply a series of GPU-based augmentations to an image tensor.
    :param image: A tensor image already on the GPU.
    :param transform_list: List of transforms to be applied to the image.
    :return: Transformed image on the GPU.
    """
    for t in transform_list:
        image = t(image)
    return image

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
        cpu_train_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
        cpu_valid_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])),
        cpu_test_list.append(v2.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225]))
    image_transforms = {
        'train': v2.Compose(cpu_train_list),
        'valid': v2.Compose(cpu_valid_list),
        'test': v2.Compose(cpu_test_list)
    }
    
    # Load data from folders
    dataset = {
        'train': AlternatingSequenceDataset(root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/train/', transform=image_transforms['train'], transition_frames=transition_frames),
        'valid': SequenceDataset(root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/val/', transform=image_transforms['valid'], transition_frames=transition_frames),
        'test': SequenceDataset(root_dir='/home/user/Documents/GitHub/medic/data/images_ts_fe_30_singles/test/', transform=image_transforms['test'], transition_frames=transition_frames)
    }

    
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
    model_ft = StrongTransitionLSTM().to(device)
    activation = nn.Softmax(dim=1)
    # class_weights = torch.tensor([1, 10]).to(device)
    criterion = nn.BCEWithLogitsLoss()#pos_weight=class_weights[1])
    optimizer = optim.AdamW(model_ft.parameters(), lr=hyper_params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyper_params['num_epochs'], eta_min=hyper_params['learning_rate']/100.)

    return model_ft, cnn_model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list


def test_model(params, hyper_params):
    since = time.time()

    model, cnn_model, dataloaders, data_sizes, criterion, optimizer, scheduler, activation, device, gpu_train_list = load_train(params, hyper_params)
    model.to(device)
    print(f"Train: {next(model.parameters()).device}")

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    logger = open(os.path.join(params['save_dir'], params['name'] + '.txt'), "w")

    for epoch in range(hyper_params['num_epochs']):
        print('Epoch {}/{}'.format(epoch, hyper_params['num_epochs'] - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid', 'test']:
            # criterion = nn.CrossEntropyLoss(weight=train_weights if phase == 'train' else valid_weights)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   


            with tqdm.tqdm(total=len(dataloaders[phase]), desc=f'{phase.capitalize()} Epoch {epoch}', unit='batch') as pbar:
                mod = 5
                i = 0

                epoch_loss = 0
                total_correct = 0
                some_one = 0
                total_samples = 0

                # Iterate over data.
                for sequences, paths, label, info in dataloaders[phase]:
                    if i % mod == 0:
                        sequences = sequences.to(device, non_blocking=True) 
                        label = label.to(device, non_blocking=True).float() 
                        # inputs = apply_gpu_transforms(inputs, gpu_train_list)
                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            # Example input to CNN
                            # sequences: [Batch, 3, Sequence, Channel, H, W]
                            B, num_parts, S, C, H, W = sequences.shape  # Batch size, sequence length, channels, height, width

                            batch_preds = []
                            batch_sequence = []
                            # batch_paths = path
                            for i in range(num_parts):  # prev, curr, next
                                seq = sequences[:, i]  # [B, S, C, H, W]
                                features, preds = cnn_model(seq.view(B * S, C, H, W))  # [B*S, F], [B*S, 3]
                                features = features.view(B, S, -1)  # [B, S, F]
                                preds = preds.view(B, S, -1)  # [B, S, 3]
                                preds_prob = F.softmax(preds, dim=-1)  # [B, S, 3] - softmax across the class dimension

                                # Get the predicted class index` (0, 1, or 2)
                                _, preds_values = torch.max(preds_prob, 2)
                                batch_preds.append(preds_values.tolist())  # [B, S, 3]

                                # Concatenate features and predictions
                                combined = torch.cat([features, preds], dim=-1)  # [B, S, F+3]
                                batch_sequence.append(combined)

                            full_sequence = torch.cat(batch_sequence, dim=-1)  # [B, S, 3*(F+3)]

                            outputs = model(full_sequence)  # [B, 1]
                            loss = criterion(outputs, label)  # [B, 1] vs [B, 1]

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                            
                            epoch_loss += loss.item()
                            
                            preds = (outputs > 0.5).float()  # Convert logits to binary predictions (0 or 1)
                            print(f"{outputs.item():.6f}", preds.item(), label.item(), batch_preds, info, paths)
                            # for s in paths:
                            #     print([p[0] for p in s])
                            
                            total_correct += torch.sum(preds == label.data)  # Count correct predictions
                            some_one += (preds.item() == 1)
                            total_samples += label.size(0)  # Increment total samples
```