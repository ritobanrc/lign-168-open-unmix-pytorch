import torch
import torchaudio
import random
from pathlib import Path
from typing import Optional, Union, Tuple, List, Any, Callable
import glob
import os
import math

from openunmix import data

major_third = 4
perfect_fourth = 5
perfect_fifth = 7
major_sixth = 9
octave = 12

intervals = [major_third, perfect_fourth, perfect_fifth, major_sixth, octave]

def gen_overlaid_data(interval, data, sr):
    shifted_data = torchaudio.functional.pitch_shift(data, sr, interval)
    overlaid_data = 0.5*(data + shifted_data)

    return (shifted_data, overlaid_data)



class ESMUC_Dataset_Isolated(data.UnmixDataset):
    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        # sample_rate: float,
        # seq_duration: Optional[float] = None,
        # source_augmentations: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root).expanduser()
        self.sample_rate = 44100
        
        pattern = "*IS*.wav"
        # Use glob to find files matching the pattern
        random.seed(0)        
        self.matching_files = glob.glob(os.path.join(self.root, pattern))
        random.shuffle(self.matching_files)
        count = len(self.matching_files)
        valid_count = math.ceil(0.1 * count)
        if split == 'train':
            self.matching_files = self.matching_files[valid_count:]
        elif split == 'valid':
            self.matching_files = self.matching_files[:valid_count]
        else:
            print("Split not recognized: ", split)

    
    def __getitem__(self, index: int) -> Any:
        file = self.matching_files[index]
        info = data.load_info(file)      
        audio, sr = data.load_audio(file)

        length = min(audio.shape[1], sr*5)
        
        audio = audio[0:1, :length]
    
        shift = random.choice(intervals)
        mix = gen_overlaid_data(shift, audio, sr)
        return mix[1], audio

    def __len__(self) -> int:
        return len(self.matching_files)

        