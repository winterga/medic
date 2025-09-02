###
# I do NOT use this yet but I will in the future. The integration of SSP class is not complete but I am using this 
# implicitly in my class files. That is, my code is running through the concepts of SSP but it is not called SSP
# explicitly. I will use this class in the future to make my code more readable and modular.
### 

# Stride-Spaced Pooling (SSP) class definition
class SSP:

    sequence_length = 0
    window_size = 0
    frame_gap = 0
    stride = 0

    def __init__(self, sequence_length: int, window_size: int, frame_gap: int, stride: int):
        print(f"Creating a Stride Spaced Pooling (SSP) object with: \\ seq_len: {sequence_length} \\ window size: {window_size} \\ frame_gap: {frame_gap} \\ stride: {stride} ")
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.frame_gap = frame_gap
        self.stride = stride
        
        assert self.sequence_length > 0, "Sequence length must be greater than 0"
        assert self.window_size > 0, "Window size must be greater than 0"
        assert self.frame_gap > 0, "Frame gap must be greater than 0"
        assert self.stride > 0, "Stride must be greater than 0"

        self.span = (window_size - 1) * frame_gap + 1
        assert sequence_length >= self.span, "Sequence length must be at least long enough for one window"

        """
        • SL: Sequence Length [0...14] = 15
        • W : Window Size [0, 5, 10] = 3
        • G: Frame Gap [0, 5, 10], [1, 6, 11] = 1; [0, 5, 10], [3, 8, 13] = 3
        • S: Stride [0, 5, 10] = 5; [0, 2, 4] = 2
        • Span: [0,1,2] = 3; [0,3,6] = 7; [1,4,7] = 7; [2,5,8] = 7; [3,6,9] = 7; [4,7,10] = 7
        """

        self.num_temporal_timesteps = (self.sequence_length - self.span) // stride + 1

        """
        The `num_timesteps` to the LSTM will be equal to `(SL - span) / (stride + 1)`
        • Num temporal timesteps = 5  # computed as (SL - span) // S + 1 = (15-11)//1 + 1
        """

    def get_windows(self):
        windows = []
        # iterate over possible starting points
        for start in range(0, self.stride):
            # Build a window according to window_size and frame_gap
            window = []
            for i in range(start, self.sequence_length, self.stride):
                window_idx = [i + j * self.frame_gap for j in range(self.window_size)]
                if max(window_idx) < self.sequence_length:
                    windows.append(window_idx)
        return windows