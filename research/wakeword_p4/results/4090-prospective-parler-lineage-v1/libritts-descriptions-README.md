---
license: cc-by-4.0
task_categories:
- text-to-speech
language:
- en
size_categories:
- 10K<n<100K

dataset_info:
- config_name: clean
  features:
  - name: text
    dtype: string
  - name: text_original
    dtype: string
  - name: speaker_id
    dtype: string
  - name: path
    dtype: string
  - name: chapter_id
    dtype: string
  - name: id
    dtype: string
  - name: snr
    dtype: float32
  - name: c50
    dtype: float32
  - name: speech_duration
    dtype: float64
  - name: speaking_rate
    dtype: string
  - name: phonemes
    dtype: string
  - name: stoi
    dtype: float64
  - name: si-sdr
    dtype: float64
  - name: pesq
    dtype: float64
  - name: gender
    dtype: string
  - name: utterance_pitch_std
    dtype: float64
  - name: utterance_pitch_mean
    dtype: float64
  - name: pitch
    dtype: string
  - name: noise
    dtype: string
  - name: reverberation
    dtype: string
  - name: speech_monotony
    dtype: string
  - name: sdr_noise
    dtype: string
  - name: pesq_speech_quality
    dtype: string
  - name: accent
    dtype: string
  - name: text_description
    dtype: string
  splits:
  - name: dev.clean
    num_bytes: 5382981.626046025
    num_examples: 5589
  - name: test.clean
    num_bytes: 4711308.860243953
    num_examples: 4689
  - name: train.clean.100
    num_bytes: 31313255.308738567
    num_examples: 32215
  - name: train.clean.360
    num_bytes: 110262720.55497913
    num_examples: 112326
  download_size: 53796229
  dataset_size: 151670266.35000768
- config_name: other
  features:
  - name: text
    dtype: string
  - name: text_original
    dtype: string
  - name: speaker_id
    dtype: string
  - name: path
    dtype: string
  - name: chapter_id
    dtype: string
  - name: id
    dtype: string
  - name: snr
    dtype: float32
  - name: c50
    dtype: float32
  - name: speech_duration
    dtype: float64
  - name: speaking_rate
    dtype: string
  - name: phonemes
    dtype: string
  - name: stoi
    dtype: float64
  - name: si-sdr
    dtype: float64
  - name: pesq
    dtype: float64
  - name: gender
    dtype: string
  - name: utterance_pitch_std
    dtype: float64
  - name: utterance_pitch_mean
    dtype: float64
  - name: pitch
    dtype: string
  - name: noise
    dtype: string
  - name: reverberation
    dtype: string
  - name: speech_monotony
    dtype: string
  - name: sdr_noise
    dtype: string
  - name: pesq_speech_quality
    dtype: string
  - name: accent
    dtype: string
  - name: text_description
    dtype: string
  splits:
  - name: dev.other
    num_bytes: 4058546.371125081
    num_examples: 4342
  - name: test.other
    num_bytes: 4335314.71640625
    num_examples: 4716
  - name: train.other.500
    num_bytes: 185984836.26363304
    num_examples: 194626
  download_size: 67735264
  dataset_size: 194378697.35116437
configs:
- config_name: clean
  data_files:
  - split: dev.clean
    path: clean/dev.clean-*
  - split: test.clean
    path: clean/test.clean-*
  - split: train.clean.100
    path: clean/train.clean.100-*
  - split: train.clean.360
    path: clean/train.clean.360-*
- config_name: other
  data_files:
  - split: dev.other
    path: other/dev.other-*
  - split: test.other
    path: other/test.other-*
  - split: train.other.500
    path: other/train.other.500-*
---


# Dataset Card for Annotated LibriTTS-R


**This dataset is an annotated version of a [filtered LibriTTS-R](https://huggingface.co/datasets/parler-tts/libritts_r_filtered) [1].** 

[LibriTTS-R](https://huggingface.co/datasets/blabble-io/libritts_r) [1] is a sound quality improved version of the [LibriTTS corpus](http://www.openslr.org/60/) which is a multi-speaker English corpus of approximately 960 hours of read English speech at 24kHz sampling rate, published in 2019. 

In the `text_description` column, it provides natural language annotations on the characteristics of speakers and utterances, that have been generated using [the Data-Speech repository](https://github.com/huggingface/dataspeech).

This dataset was used alongside its original version [LibriTTS-R](https://huggingface.co/datasets/blabble-io/libritts_r) and the [English subset of MLS](https://huggingface.co/datasets/parler-tts/mls_eng) to train [Parler-TTS [Mini v1]((https://huggingface.co/parler-tts/parler-tts-mini-v1)) and [Large v1](https://huggingface.co/parler-tts/parler-tts-large-v1).
A training recipe is available in [the Parler-TTS library](https://github.com/huggingface/parler-tts).


## Motivation

This dataset is a reproduction of work from the paper [Natural language guidance of high-fidelity text-to-speech with synthetic annotations](https://www.text-description-to-speech.com) by Dan Lyth and Simon King, from Stability AI and Edinburgh University respectively. 
It was designed to train the Parler-TTS [Mini v1]((https://huggingface.co/parler-tts/parler-tts-mini-v1)) and [Large v1](https://huggingface.co/parler-tts/parler-tts-large-v1) models.

Contrarily to other TTS models, Parler-TTS is a **fully open-source** release. All of the datasets, pre-processing, training code and weights are released publicly under permissive license, enabling the community to build on our work and develop their own powerful TTS models.
Parler-TTS was released alongside:
* [The Parler-TTS repository](https://github.com/huggingface/parler-tts) - you can train and fine-tuned your own version of the model.
* [The Data-Speech repository](https://github.com/huggingface/dataspeech) - a suite of utility scripts designed to annotate speech datasets.
* [The Parler-TTS organization](https://huggingface.co/parler-tts) - where you can find the annotated datasets as well as the future checkpoints.


## Usage

Here is an example on how to oad the `clean` config with only the `train.clean.360` split.

```py
from datasets import load_dataset

load_dataset("parler-tts/libritts-r-filtered-speaker-descriptions", "clean", split="train.clean.100")
```

Streaming is also supported.

```py
from datasets import load_dataset

load_dataset("parler-tts/libritts-r-filtered-speaker-descriptions", "clean", streaming=True)
```

**Note:** This dataset doesn't actually keep track of the audio column of the original version. You can merge it back to the original dataset using [this script](https://github.com/huggingface/dataspeech/blob/main/scripts/merge_audio_to_metadata.py) from Parler-TTS or, even better, get inspiration from [the training script](https://github.com/huggingface/parler-tts/blob/main/training/run_parler_tts_training.py) of Parler-TTS, that efficiently process multiple annotated datasets.

### Dataset Description

- **License:** CC BY 4.0

### Dataset Sources

- **Homepage:** https://www.openslr.org/141/
- **Paper:** https://arxiv.org/abs/2305.18802


## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

```
@ARTICLE{Koizumi2023-hs,
  title         = "{LibriTTS-R}: A restored multi-speaker text-to-speech corpus",
  author        = "Koizumi, Yuma and Zen, Heiga and Karita, Shigeki and Ding,
                   Yifan and Yatabe, Kohei and Morioka, Nobuyuki and Bacchiani,
                   Michiel and Zhang, Yu and Han, Wei and Bapna, Ankur",
  abstract      = "This paper introduces a new speech dataset called
                   ``LibriTTS-R'' designed for text-to-speech (TTS) use. It is
                   derived by applying speech restoration to the LibriTTS
                   corpus, which consists of 585 hours of speech data at 24 kHz
                   sampling rate from 2,456 speakers and the corresponding
                   texts. The constituent samples of LibriTTS-R are identical
                   to those of LibriTTS, with only the sound quality improved.
                   Experimental results show that the LibriTTS-R ground-truth
                   samples showed significantly improved sound quality compared
                   to those in LibriTTS. In addition, neural end-to-end TTS
                   trained with LibriTTS-R achieved speech naturalness on par
                   with that of the ground-truth samples. The corpus is freely
                   available for download from
                   \textbackslashurl\{http://www.openslr.org/141/\}.",
  month         =  may,
  year          =  2023,
  copyright     = "http://creativecommons.org/licenses/by-nc-nd/4.0/",
  archivePrefix = "arXiv",
  primaryClass  = "eess.AS",
  eprint        = "2305.18802"
}
```

```
@misc{kawamura2024librittspcorpusspeakingstyle,
      title={LibriTTS-P: A Corpus with Speaking Style and Speaker Identity Prompts for Text-to-Speech and Style Captioning}, 
      author={Masaya Kawamura and Ryuichi Yamamoto and Yuma Shirahata and Takuya Hasumi and Kentaro Tachibana},
      year={2024},
      eprint={2406.07969},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2406.07969}, 
}
```


```
@misc{lacombe-etal-2024-dataspeech,
  author = {Yoach Lacombe and Vaibhav Srivastav and Sanchit Gandhi},
  title = {Data-Speech},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ylacombe/dataspeech}}
}
```

```
@misc{lyth2024natural,
      title={Natural language guidance of high-fidelity text-to-speech with synthetic annotations},
      author={Dan Lyth and Simon King},
      year={2024},
      eprint={2402.01912},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```