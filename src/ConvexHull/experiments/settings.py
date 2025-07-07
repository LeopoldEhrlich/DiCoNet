from dataclasses import dataclass

@dataclass
class DCN_Settings:
    num_examples_test: int = 256
    batch_size: int = 32
    path_dataset: str = './data'
    path: str = './models/reg'
    
    dynamic: bool = True

    load_merge: str = './models/reg'
    num_units_merge: int = 512
    rnn_layers: int = 1
    grad_clip_merge: float = 2.0
    merge_sample: bool = False

    load_split: str = './models/reg'
    split_layers: int = 5
    num_units_split: int = 15
    grad_clip_split: float = 40.0
    regularize_split: bool = False
    beta: float = 1.0
    random_split: bool = False