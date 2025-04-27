# Dog vs Cat Classifier

This project implements a **binary image classification** system to distinguish between dog and cat images, built using **PyTorch** with a **ConvNeXtV2** backbone for high-performance feature extraction.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Checkpoints](#checkpoints)
- [Configuration](#configuration)
- [Results](#results)
- [License](#license)

---

## Overview
This repository contains a deep learning pipeline for the **Dog vs Cat classification** task, optimized with techniques such as **early stopping**, **checkpoint saving**, and **GPU acceleration**.  
It is designed for easy extension to other binary or multi-class image classification tasks.

---

## Features
- **ConvNeXtV2** as a high-performance, pre-trained feature extractor.
- **Binary classifier** (dog or cat) via a fully connected head.
- **Early stopping** based on validation loss to prevent overfitting.
- **Automatic checkpointing**: best model is saved during training.
- **Comprehensive evaluation**: loss and accuracy are reported per epoch.
- **GPU-accelerated** training (auto-detects CUDA if available).

---

## Dataset
We use the [Cat and Dog dataset from Kaggle](https://www.kaggle.com/datasets/tongpython/cat-and-dog).

**Expected directory structure:**
<<<<<<< HEAD
dataset/ ├── training_set/ │ ├── cats/ │ └── dogs/ └── test_set/ ├── cats/ └── dogs/

=======
```
dataset/
├── training_set/ training_set/
│   ├── cats/
│   └── dogs/
└── test_set/ test_set/
    ├── cats/
    └── dogs/
```
>>>>>>> 4372d81adcafd39cb4a242ffb96fca36752bd9ec

---

## Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/dog-vs-cat-classifier.git
cd dog-vs-cat-classifier
pip install -r requirements.txt
---

## Training
<<<<<<< HEAD
To start training, run the following command:

```bash
python train.py --data_dir <path_to_dataset>
```

## Training
=======
To start training, simply run:

```bash
python train.py
```

The dataset path is already set to `data/training_set` in the script.

>>>>>>> 4372d81adcafd39cb4a242ffb96fca36752bd9ec
- The best model will be saved in the `checkpoints/` directory.
- The final model will be stored as `checkpoints/catdog_final.pth`.

---

## Evaluation
After each epoch, the model is evaluated on the validation set. The following metrics are reported:
- **Validation Loss**
- **Validation Accuracy**

You can adjust the evaluation frequency or metrics in `train.py`.

---

## Checkpoints
- **Best model**: Saved as `checkpoints/best_model.pth`.
- **Final model**: Saved as `checkpoints/catdog_final.pth`.

---

## Configuration
You can modify hyperparameters directly in `train.py`:
- `batch_size`
- `learning_rate`
- `num_epochs`
- `early_stopping_patience`

For larger experiments, consider moving hyperparameters into a YAML or JSON config file.

---

## Results
Example results after 10 epochs (with `batch_size=32` and `learning_rate=1e-5`):

| Metric             | Score   |
|--------------------|---------|
| **Validation Loss** | 0.0179 |
<<<<<<< HEAD
| **Validation Accuracy** | 96.8% |

Note: Results may vary depending on the random seed and hardware.
=======
| **Validation Accuracy** | 99.5% |

Note: Results may vary depending on the random seed and hardware.
>>>>>>> 4372d81adcafd39cb4a242ffb96fca36752bd9ec
