
# Chest X-Ray Pneumonia Classification

## Project Overview

This project builds a classification model for detecting pneumonia from medical chest x-ray images using Deep Learning with ResNet50V2. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and processed using AWS SageMaker for scalable cloud-based training and deployment. The system uses transfer learning with a pre-trained ResNet50V2 convolutional neural network, which has been fine-tuned for binary classification of pneumonia versus normal X-rays.

![X-Ray Image](https://oryon.co.uk/app/uploads/2024/02/shutterstock_182657084-1-1-scaled.jpg)

### Objectives
- Implement a deep learning model for pneumonia classification
- Demonstrate transfer learning using ResNet50V2
- Document a complete ML workflow from data preparation to evaluation
- Provide a trained model ready for deployment

## Data Flow and Processing Pipeline
The project follows a standard machine learning workflow with specific adaptations for medical image processing:

![workflow](https://github.com/user-attachments/assets/d35c5cfd-3d1c-4440-9626-1d2d0af0c1db)


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

### Summary
| Step               | Description                  | Implementation                                      |
|--------------------|------------------------------|-----------------------------------------------------|
| Dataset Source     | Chest X-ray images from Kaggle | Uploaded to S3 bucket                               |
| Data Extraction    | Unzipping dataset             | Extracted to SageMaker `/tmp/` directory            |
| Data Structure     | Pre-organized folders         | `train`, `val`, and `test` directories              |
| Normalization      | Pixel value scaling           | `ImageDataGenerator` with `rescale=1./255`          |
| Data Augmentation  | Increased training diversity  | Horizontal flips and zoom via `ImageDataGenerator`  |
| Batch Processing   | Efficient loading             | Batch size of 128 for training                      |

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
 
### Model Design Summary:
- Used GlobalAveragePooling2D instead of Flatten to reduce parameters and prevent overfitting
- Added Dropout (0.5) for regularization during training
- Binary classification with sigmoid activation for pneumonia probability
- Adam optimizer with Binary Crossentropy loss function

## Training 

### Training Infrastructure

![traininginfra](https://github.com/user-attachments/assets/3289ed81-e844-4cf9-bd1d-df1e471ae70d)

### Training Configuration

| Parameter        | Value            | Purpose                                               |
|------------------|------------------|--------------------------------------------------------|
| Epochs           | 5                | Balance between training time and accuracy            |
| Batch Size       | 128              | Optimized for GPU memory and throughput               |
| Optimizer        | Adam             | Adaptive learning rate optimization                   |
| Loss Function    | Binary Crossentropy | Standard for binary classification                |
| Early Stopping   | Enabled          | Prevent overfitting by monitoring validation loss      |
| Training Time    | ~3 min/epoch     | Total training time ~15 minutes                        |




## Performance Results
The trained model achieved strong results on the pneumonia classification task:

| Metric                    | Training | Validation | Test  |
|---------------------------|----------|------------|-------|
| Accuracy                  | 91.2%    | 87.5%      | -     |
| Normal Class Precision    | -        | -          | 0.75  |
| Normal Class Recall       | -        | -          | 0.38  |
| Pneumonia Class Precision | -        | -          | 0.58  |
| Pneumonia Class Recall    | -        | -          | 0.88  |

### Key Performance Insights:

- High recall (0.88) for pneumonia cases indicates good sensitivity for detecting disease
- Lower recall for normal cases suggests tendency toward false positives
- Model prioritizes identifying potential pneumonia cases (fewer false negatives)
- Performance balance could be improved through further hyperparameter tuning to balance Normal case detection


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


