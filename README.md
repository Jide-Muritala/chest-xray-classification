
# Chest X-Ray Pneumonia Classification

## Project Overview

This goal of this project is to build an AI model for detecting pneumonia from chest x-rays using Deep Learning with ResNet50. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and processed using AWS SageMaker for scalable cloud-based training and deployment.

![X-Ray Image](https://oryon.co.uk/app/uploads/2024/02/shutterstock_182657084-1-1-scaled.jpg)

### AWS SageMaker Specs
- **Instance Type:** `ml.g4dn.xlarge`
- **NVIDIA T4, 4 vCPUs, 16 GiB RAM**
- **Framework:** TensorFlow/Keras


## Data Processing Steps
### 1. **Dataset Preparation**
- The dataset was uploaded to the connected Amazon S3 bucket.

### 2. **Data Loading and Preprocessing**
- Extracted the zipped dataset from S3 to SageMaker’s `/tmp/` directory.
- The unzipped dataset was already structured into `train`, `val`, and `test` folders.
- Used `ImageDataGenerator` for:
  - **Rescaling:** Normalized pixel values (`1./255`).
  - **Augmentation:** Applied horizontal flips and zoom for diversity.
  - **Batch Processing:** Efficiently loaded images during training.

### 3. **Visualization**
- Plotted data distribution using Seaborn.
- Displayed sample images to verify dataset correctness.

## Model Architecture
- **Base Model:** `ResNet50V2` (Pre-trained on ImageNet, `include_top=False`).
- **Load Pretrained ResNet50V2 (without fully connected layer):**
  - Used GlobalAveragePooling2D() instead of Flatten()
    - GlobalAveragePooling2D() is better than Flatten() for CNNs. Flatten() produces too many parameters, which increases memory usage and slows training. GlobalAveragePooling2D() reduces dimensions while keeping features intact. Hence, faster training and less overfitting
  - Dense (128 units, ReLU activation, Dropout 0.5)
  - Output Layer (1 unit, Sigmoid activation)
- **Training Configuration:**
  - **Optimizer:** Adam
  - **Loss Function:** Binary Crossentropy
  - **Metrics:** Accuracy

## Training Process
- **Epochs:** 5 (Balanced between speed and accuracy)
- **Batch Size:** 128 (Efficient for SageMaker instance)
- **Early Stopping:** Used to prevent overfitting
- **Training Time:** ~3 minutes per epoch on AWS SageMaker

## Model Evaluation
- **Final Accuracy:** ~91.2% (Training), 87.5% (Validation)
- **Precision & Recall:**
  - Normal: Precision (0.75), Recall (0.38)
  - Pneumonia: Precision (0.58), Recall (0.88)
- **Insights:**
  - Model is more sensitive to Pneumonia cases (high recall).
  - Needs further tuning to balance Normal case detection.

## Deployment
- Model saved in **Keras format (`model.keras`)**.

## Future Improvements
- **Use a larger GPU instance** for faster training.
- **Experiment with different architectures** (EfficientNet, MobileNet).
- **Hyperparameter tuning** to improve classification balance.

## How to Reproduce
1. Clone the repository:
   ```sh
   git clone https://github.com/Jide-Muritala/chest-xray-classification.git
   ```
2. Run in **AWS SageMaker Jupyter Notebook**:
   ```sh
   python chest-xray.ipynb
   ```

---


