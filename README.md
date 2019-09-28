# Recursion Cellular Image Classification

This is an another one approach to solve the competition from kaggle
[Recursion Cellular Image Classification](https://www.kaggle.com/c/recursion-cellular-image-classification).

44th place out of 263 (silver medal) with 0.95340 accuracy score (top 1 -- 0.99763).

### Prerequisites

- [NVidia apex](https://github.com/NVIDIA/apex) (optional)
- [rxrx1-utils](https://github.com/recursionpharma/rxrx1-utils)

```bash
pip install -r requirements.txt
```

### Usage

First download the train and test data from the competition link.

To train the model run

```bash
bash ./run.sh
```

This will generates trained models and submission file.

### Approach

Detailed solution see in
[this presentation](presentation/Recursion_Cellular_Image_Classification.pdf).
