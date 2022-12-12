## Towards Autoformalization of Mathematics and Code Correctness: Experiments with Elementary Proofs

**Dependencies:**
- Python 3
- PyTorch (w/ CUDA)
- Numpy
- Coq 8.13.* w/ coqtop (to evaluate generated proofs, see `data/README.md`)
- Programming Language Foundations (included in `data/plf/`)

**How to Run:**
- Ensure that all dependencies are installed (see above).
- Run `python3 translate.py` with relevant arguments. These are as follows:

| **Command** | **Notes** |
|--|--|
| `--dataset_path=path/to/data/folder` | Path of the dataset directory containing subfolders to training/test/etc. |
| `--num_epochs=<int>` | Number of epochs to run training for. Default: 200 |
| `--batch_size=<int>` | Batch size for training. Default: 10 |
| `--learning_rate=<float>` | Initial learning rate for Adam. Default: 0.001 |
| `--N=<int>` | Stacked layers of encoders/decoders. Default: 4 |
| `--T=<int>` | Recursive passes through encoder/decoder stack. Default: 4 |
| `--d_model=<int>` | Size of the encoder/decoder states. Default: 64 |
| `--d_ff=<int>` | Width of the feed-forward layers. Default: 512 |
| `--H=<int>` | Number of attention heads. Default: 16 |
| `--k=<int>` | Width of relative self-attention clipping. Default: 2 |
| `--dropout=<float>` | Dropout probability. Default: 0.25 |
| `--alpha=<float>` | Label smoothing rate. Default: 0.0 |
| `--test=test_set` | Puts program into testing mode (skips training). Takes the name of the subfolder to do testing with as an argument. Pairs with `--dataset_path` to get the full path to data. Default: test |
| `--subset=[0\|1\|2]` | Break data into [0] Full, [1] Theorems Only, or [2] Proofs Only. Default: 0 |
| `--load` | Resume training by loading the `.pt` file specified by `--model`. |
| `--model=model_name` | Name of the `.pt` file (excluding `.pt`) the model will be saved to/loaded from. Default: ltc |
| `--scheduler=[step\|exponential\|plateau]` | Learning rate scheduler. Default: step |
| `--step=<int>` | Step size for StepLR learning rate scheduler. Default: 25 |
| `--gamma=<float>` | Gamma value for ExponentialLR learning rate scheduler. Default: 0.9 |
| `--patience=<int>` | Patience size for ReduceLROnPlateau learning rate scheduler. Default: 10 |
Example commands:
- To train on arithmetic data: `python3 translate.py --dataset_path=../data/arithmetic --model=arith`
- To test on handwritten arithmetic data: `python3 translate.py --dataset_path=../data/arithmetic --model=arith --test=handwritten`

**Code Structure**
- `translate.py`: Main file including routines for training and test evaluation.
- `load.py`: Preprocessing of input files/data.
- `model.py`: Implementation of large architectural features, e.g. encoder, decoder, copy mechanism, etc.
- `attention.py`: Implementation of attention mechmanisms, including self-attention with relative positional embeddings.
- `util.py`: Implementation of various utility routines and functions, e.g. label smoothing, loss functions, example batching, etc.
