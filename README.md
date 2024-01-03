# U-Net for Vessel Segmentation

## Overview

This repository contains the implementation and application of the U-Net convolutional neural network architecture for the specific task of vessel segmentation in medical images. The U-Net architecture is renowned for its effectiveness in segmentation tasks, and here, it has been configured and trained to accurately extract the morphology of blood vessels in various types of medical images, providing a valuable tool for advanced medical analyses.

## Key Features

1. **U-Net Architecture:**
   - The repository includes the implementation of the U-Net architecture, a convolutional neural network specially designed for segmentation tasks. The symmetrical structure of U-Net, with encoding and decoding paths, preserves fine details in segmentations.

2. **Custom Training:**
   - The provided code allows training the U-Net using specific datasets for medical images, ensuring the adaptation of the network to the particular characteristics of blood vessels.

3. **Performance Evaluation:**
   - Evaluation metrics such as accuracy, recall, F1-score, and possibly task-specific metrics for vessel segmentation are integrated into the repository to measure the network's performance on test data.

4. **Results Visualization:**
   - Tools are included to visualize segmentation results, facilitating qualitative interpretation of the U-Net's performance.

5. **Detailed Documentation:**
   - Comprehensive documentation accompanies the code, explaining the U-Net configuration, training details, and clear instructions on how to use the trained model for segmentations on new images.

This repository is a valuable contribution to the community interested in deep learning applications in medicine, offering a robust, ready-to-use implementation of U-Net for vessel segmentation. The code's flexibility allows extension to other medical segmentation tasks or customization for different datasets, establishing itself as a versatile and effective tool for the analysis of medical images.

## Getting Started

Follow these instructions to get started with using the U-Net for vessel segmentation on your own datasets.

### Prerequisites

- CUDA Version: 12.2
- Python 3.10
- tensorflow
- opencv-python
- numpy
- scikit-learn

### Installation

### Installation

1. **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

2. **Activate the virtual environment:**
    - On Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    - On Unix or MacOS:
        ```bash
        source venv/bin/activate
        ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Run the neural network script:**
    ```bash
    python3 rede_neural.py
    ```

