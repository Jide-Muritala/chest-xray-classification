# Chest X-Ray Pneumonia Classification

## Project Overview

This project builds a classification model for detecting pneumonia from medical chest x-ray images using Deep Learning with ResNet50V2. The dataset was sourced from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and processed using AWS SageMaker for scalable cloud-based training and deployment. The system uses transfer learning with a pre-trained ResNet50V2 convolutional neural network, which has been fine-tuned for binary classification of pneumonia versus normal X-rays. The model is implemented using TensorFlow/Keras, which provides high-level APIs for building and training neural networks.

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

The model was configured with specific hyperparameters optimized for the binary classification task of pneumonia detection:

| Hyperparameter   | Value              | Purpose                                                   |
|------------------|--------------------|-----------------------------------------------------------|
| Optimizer        | Adam               | Adaptive learning rate optimization                       |
| Loss Function    | Binary Crossentropy| Standard loss for binary classification problems          |
| Metrics          | Accuracy           | Primary evaluation metric during training                 |
| Batch Size       | 128                | Optimized for GPU memory and throughput                   |
| Epochs           | 5                  | Balanced between training time and accuracy               |
| Early Stopping   | Enabled            | Prevent overfitting by monitoring validation loss         |

The batch size of 128 was specifically chosen to maximize efficiency on the SageMaker GPU instance, while the moderate epoch count of 5 balanced training time against model accuracy.

## Performance Results

The training process yielded the following key performance indicators:

- Training Time: ~3 minutes per epoch
- Final Training Accuracy: 91.2%
- Final Validation Accuracy: 87.5%

The trained model achieved strong results on the pneumonia classification task:

| Class     | Precision | Recall |
|-----------|-----------|--------|
| Normal    | 0.75      | 0.38   |
| Pneumonia | 0.58      | 0.88   |

1. Pneumonia Detection (Positive Class):

- High recall (0.88): Successfully identifies 88% of all pneumonia cases
- Lower precision (0.58): 42% of predicted pneumonia cases are false positives

2. Normal Detection (Negative Class):

- High precision (0.75): When the model predicts normal, it's correct 75% of the time
- Low recall (0.38): The model misses 62% of normal cases, classifying them as pneumonia

This imbalance suggests the model has been optimized to minimize false negatives for pneumonia cases, which is often desirable in medical diagnostics where missing a disease is typically more problematic than a false alarm.


### Key Performance Insights:

#### Key Strengths
1. **High Sensitivity for Pneumonia**: The model successfully identifies 88% of pneumonia cases, making it effective for screening purposes where catching cases is critical.

2. **Good Overall Accuracy**: The 87.5% validation accuracy indicates the model performs well above random chance and could be valuable in clinical support settings.

#### Key Limitations
1. **Normal Case Detection**: The model's low recall (0.38) for normal cases means it frequently misclassifies healthy patients as having pneumonia, which could lead to unnecessary follow-up procedures.

2. **Precision-Recall Tradeoff**: The current model configuration favors recall over precision for pneumonia cases, which may be appropriate for a screening tool but less suitable for definitive diagnosis.

## Usage and Deployment
The repository provides a trained model (`model.keras`) that can be used for inference on new chest X-ray images. 

Basic Usage Steps:

1. Clone the repository
 ```sh
   git clone https://github.com/Jide-Muritala/chest-xray-classification.git
   ```
2. Load the pre-trained model.keras file
3. Preprocess new X-ray images (resize to 220x220, normalize)
4. Run inference to get pneumonia probability
5. Apply threshold to determine classification

## Future Improvements
1. Threshold Adjustment: Experimenting with different classification thresholds could help balance precision and recall based on specific clinical needs.

2. Class Weighting: Implementing class weights during training could help address the imbalance in normal vs. pneumonia detection.

3. Architectural Improvements: Exploring alternative architectures like EfficientNet or MobileNet might yield better performance characteristics.

4. Hyperparameter Tuning: Systematic hyperparameter optimization could improve the balance between normal and pneumonia case detection.

These improvements aim to enhance the model's overall performance while maintaining its high sensitivity for pneumonia detection[.](https://deepwiki.com/Jide-Muritala/chest-xray-classification)


---


