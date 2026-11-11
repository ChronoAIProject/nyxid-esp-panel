---
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: timestamps_start
    sequence: float64
  - name: timestamps_end
    sequence: float64
  - name: speakers
    sequence: string
  splits:
  - name: dev
    num_bytes: 2338411143
    num_examples: 216
  - name: test
    num_bytes: 5015872396
    num_examples: 232
  download_size: 7296384603
  dataset_size: 7354283539
configs:
- config_name: default
  data_files:
  - split: dev
    path: data/dev-*
  - split: test
    path: data/test-*
tags:
- speaker diarization
- voice activity detection
license: cc-by-4.0
language:
- en
---


# Dataset Card for the Voxconverse dataset 

VoxConverse is an audio-visual diarisation dataset consisting of multispeaker clips of human speech, extracted from YouTube videos. Updates and additional information about the dataset can be found on the [dataset website](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/index.html).

Note: This dataset has been preprocessed using [diarizers](https://github.com/huggingface/diarizers/tree/main/datasets). It makes the dataset compatible with diarizers to fine-tune [pyannote](https://huggingface.co/pyannote/segmentation-3.0) segmentation models.


# Example Usage

```
from datasets import load_dataset
ds = load_dataset("diarizers-community/voxconverse")

print(ds)
```

gives: 

```
DatasetDict({
    train: Dataset({
        features: ['audio', 'timestamps_start', 'timestamps_end', 'speakers'],
        num_rows: 136
    })
    validation: Dataset({
        features: ['audio', 'timestamps_start', 'timestamps_end', 'speakers'],
        num_rows: 18
    })
    test: Dataset({
        features: ['audio', 'timestamps_start', 'timestamps_end', 'speakers'],
        num_rows: 16
    })
})
```

# Dataset source

- Homepage: https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- Repository: https://github.com/joonson/voxconverse?tab=readme-ov-file
- Preprocessed using [diarizers](https://github.com/kamilakesbi/diarizers/tree/main/datasets)


# Citation

```
@article{chung2020spot,
  title={Spot the conversation: speaker diarisation in the wild},
  author={Chung, Joon Son and Huh, Jaesung and Nagrani, Arsha and Afouras, Triantafyllos and Zisserman, Andrew},
  booktitle={Interspeech},
  year={2020}
}
```

# Contribution 

Thanks to [@kamilakesbi](https://huggingface.co/kamilakesbi) and [@sanchit-gandhi](https://huggingface.co/sanchit-gandhi) for adding this dataset.





