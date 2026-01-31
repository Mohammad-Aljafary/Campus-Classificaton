# Campus Classification

A deep learning image classification project using PyTorch and ResNet-18 to classify campus images into different categories.

## Overview

This project implements a transfer learning approach using a pre-trained ResNet-18 model for campus image classification. The model is fine-tuned on a custom dataset with data augmentation techniques to improve generalization.

## Features

- **Transfer Learning**: Utilizes pre-trained ResNet-18 model from torchvision
- **Data Augmentation**: Implements various augmentation techniques using Albumentations library:
  - Resize and cropping
  - Horizontal flipping
  - Random rotation
  - Blur
- **Class Imbalance Handling**: Uses weighted loss function to address class imbalance
- **GPU Support**: Automatically detects and uses CUDA-enabled GPU if available
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Visualization**: Training and validation loss/accuracy curves, confusion matrix, and classification report

## Project Structure

```
Campus-Classificaton/
├── model2.ipynb         # Main Jupyter notebook with training pipeline
├── best_model.pth       # Saved best model weights
├── requirements.txt     # Project dependencies
├── datasets/            # Dataset directory (not included)
└── README.md            # This file
```

## Requirements

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Dependencies

- torch
- torchvision
- albumentations
- matplotlib
- numpy
- ipykernel
- scikit-learn

## Dataset Structure

Organize your dataset in the following structure:

```
datasets/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   └── ...
└── ...
```

The dataset is automatically split into:
- **Training set**: 80%
- **Validation set**: 10%
- **Test set**: 10%

## Usage

1. **Prepare your dataset**: Place your images in the `datasets/` directory following the structure above

2. **Open the notebook**:
   ```bash
   jupyter notebook model2.ipynb
   ```

3. **Run all cells**: Execute cells sequentially to:
   - Load and preprocess the dataset
   - Set up data augmentation
   - Configure the ResNet-18 model
   - Train the model (default: 10 epochs)
   - Evaluate on test set
   - Visualize training metrics

## Model Architecture

- **Base Model**: ResNet-18 (pre-trained on ImageNet)
- **Modification**: Final fully connected layer replaced to match number of classes
- **Training Strategy**:
  - All layers frozen except the final classification layer
  - Fine-tuning on custom dataset
  - Uses AdamW optimizer with learning rate 0.001

## Data Preprocessing

### Training Data
- Resize to 256×256
- Random crop to 224×224
- Horizontal flip (p=0.5)
- Random rotation (±15°)
- Blur (p=0.2)
- ImageNet normalization

### Validation/Test Data
- Resize to 256×256
- Center crop to 224×224
- ImageNet normalization

## Training Configuration

- **Batch Size**: 32
- **Epochs**: 10
- **Optimizer**: AdamW
- **Learning Rate**: 0.001
- **Loss Function**: Cross-Entropy with class weights
- **Device**: CUDA (GPU) if available, otherwise CPU

## Results

The model saves the best performing checkpoint based on validation accuracy. After training, the notebook displays:
- Training loss per epoch
- Validation accuracy and loss per epoch
- Final test accuracy
- Loss curves visualization
- Confusion matrix and classification report

## Notes

- The project handles class imbalance using sklearn's `compute_class_weight`
- Automatic device selection (GPU/CPU) with fallback mechanism
- Random seed (42) set for reproducibility

## License

This project is for educational purposes as part of university coursework.

## Author

University Project - Campus Classification