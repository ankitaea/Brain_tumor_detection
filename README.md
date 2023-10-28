## ** Project Overview

This project implements a Convolutional Neural Network (CNN) for brain tumor detection using medical images. The dataset consists of brain images categorized into classes based on the presence or absence of tumors. The model is built using the TensorFlow and Keras frameworks. The architecture utilizes a pre-trained ResNet50V2 model for feature extraction.

## ** Files and Directories

- **Utils.ipynb**: Jupyter Notebook containing utility functions for data preprocessing, model training, and visualization.
- **BTD_001.zip**: Dataset containing brain images categorized into classes.
- **Readme.md**: Project readme file explaining the project overview, file structure, and usage instructions.

## ** Steps to Run the Code

### 1. **Data Preparation**:
   - The `BTD_001.zip` file contains the dataset. Unzip the file to the specified directory in the notebook.
   - The dataset is organized into classes, and the class distribution is visualized using pie charts.

### 2. **Image Preprocessing**:
   - Images are resized to (630, 630) and cropped around the brain contour for focus.
   - Data augmentation techniques, such as rotation, shifting, and flipping, are applied for increased variability.

### 3. **Model Architecture**:
   - The base model is ResNet50V2, pre-trained on ImageNet.
   - The top layers consist of Global Average Pooling and Dense layers with softmax activation for classification.
   - The model is compiled with sparse categorical cross-entropy loss and Adam optimizer.

### 4. **Model Training**:
   - The model is trained using the prepared data, and training/validation loss and accuracy are plotted over epochs.

### 5. **Feature Extraction**:
   - Features are extracted using the pre-trained ResNet50V2 model.
   - Support Vector Machine (SVM) is trained on the extracted features for classification.

### 6. **Evaluation**:
   - Model performance is evaluated using the test set, and accuracy is displayed.
   - Predictions are visualized using sample images from the validation set.

## ** Requirements

- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Plotly
- Scikit-learn

## ** Usage

1. Clone the repository: `git clone <repository-url>`<br>
2. Navigate to the project directory: `cd <repository-folder>`<br>
3. Ensure all required libraries are installed using: `pip install -r requirements.txt` <br>
4. Run `Utils.ipynb` in a Jupyter Notebook environment for data preprocessing, model training, and evaluation. <br>
