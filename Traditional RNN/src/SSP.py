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

    def __init__(self, sequence_length, window_size, frame_gap, stride):
        self.sequence_length = sequence_length
        self.window_size = window_size
        self.frame_gap = frame_gap
        self.stride = stride
        
        assert self.sequence_length > 0, "Sequence length must be greater than 0"
        assert self.window_size > 0, "Window size must be greater than 0"
        assert self.frame_gap > 0, "Frame gap must be greater than 0"
        assert self.stride > 0, "Stride must be greater than 0"
        
        assert sequence_length % window_size == 0, "Sequence length must be a multiple of window size"
        assert sequence_length % frame_gap == 0, "Sequence length must be a multiple of frame gap"
        assert sequence_length % stride == 0, "Sequence length must be a multiple of stride"
        assert sequence_length % stride * window_size, "Sequence length must be a multiple of stride * window size"

        """
        • SL: Sequence Length [0...14] = 15
        • W : Window Size [0, 5, 10] = 3
        • G: Frame Gap [0, 5, 10], [1, 6, 11] = 1; [0, 5, 10], [3, 8, 13] = 3
        • S: Stride [0, 5, 10] = 5; [0, 2, 4] = 2
        """