\# Handwritten Digit Recognition using MNIST Dataset



\## Project Overview



This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0-9) from the MNIST dataset available on Kaggle. The model achieves high accuracy (~99%) in digit classification.



\## Problem Statement



In the modern digital era, recognizing handwritten digits has numerous applications, such as:

\- Digitizing bank cheques

\- Reading postal codes

\- Developing assistive technologies

\- OCR (Optical Character Recognition) systems



\*\*Question\*\*: Can we build a machine learning model that can correctly identify handwritten digits (0–9) from images?



\## Dataset



\- \*\*Source\*\*: Kaggle Digit Recognizer Competition

\- \*\*Training Data\*\*: `train.csv` - 42,000 images of digits (28×28 pixels)

\- \*\*Test Data\*\*: `test.csv` - 28,000 images without labels

\- \*\*Format\*\*: Flattened 28×28 grayscale images (784 pixel values per row)



\## Technologies Used



\- \*\*Python\*\* - Main programming language

\- \*\*Pandas \& NumPy\*\* - Data loading and manipulation

\- \*\*Matplotlib \& Seaborn\*\* - Data visualization

\- \*\*Scikit-learn\*\* - Train/validation split and metrics

\- \*\*TensorFlow/Keras\*\* - CNN model building and training



\## Model Architecture



The CNN model includes:



1\. \*\*Convolutional Layers\*\*: Extract image features

&nbsp;  - Conv2D layers with 32, 64, and 128 filters

&nbsp;  - Batch normalization for stable training

&nbsp;  - ReLU activation functions



2\. \*\*Pooling Layers\*\*: Reduce dimensionality

&nbsp;  - MaxPooling2D layers



3\. \*\*Regularization\*\*: Prevent overfitting

&nbsp;  - Dropout layers (0.25-0.5)

&nbsp;  - Early stopping

&nbsp;  - Learning rate reduction



4\. \*\*Dense Layers\*\*: Final classification

&nbsp;  - Fully connected layers (512, 256 neurons)

&nbsp;  - Softmax output for 10 digit classes



\## Project Structure



