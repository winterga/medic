# Traditional RNN

## Purpose

The Traditional RNN, currently implemented as an LSTM, uses time-series modeling to predict transitions in videos. 

## TODO

Recently, I was refactoring code in order to run experiments with different SSP values. However, two problems currently exist:

1. Runtime error for `SequentialDataset`
```
Valid Epoch 0:   0%|                                                                                                                            | 0/2204 [00:00<?, ?batch/s]/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [64,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [65,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [66,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [67,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [68,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [69,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [70,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [71,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [72,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [73,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/opt/conda/conda-bld/pytorch_1724789116784/work/aten/src/ATen/native/cuda/IndexKernel.cu:92: operator(): block: [148,0,0], thread: [74,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
RuntimeError: DataLoader worker (pid 2603631) is killed by signal: Aborted. 
/home/user/miniconda3/envs/resid-frag/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d ')
Aborted (core dumped)
```

More to be looked at here

2. The transitions went from 840 to 250 for train. This doesn't quite make sense as it should be set up in the exact same way.

Both of these problems correlate to changing the datasets themselves, I believe. 