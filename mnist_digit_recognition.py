"""
Handwritten Digit Recognition using MNIST Dataset
================================================

This project implements a Convolutional Neural Network (CNN) to recognize 
handwritten digits (0-9) from the MNIST dataset available on Kaggle.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for scripts/CI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

def load_data():
    """
    Load the MNIST dataset from CSV files
    """
    print("Loading MNIST dataset...")
    
    # Load training data
    train_data = pd.read_csv('train.csv')
    print(f"Training data shape: {train_data.shape}")
    
    # Load test data
    test_data = pd.read_csv('test.csv')
    print(f"Test data shape: {test_data.shape}")
    
    return train_data, test_data

def preprocess_data(train_data, test_data):
    """
    Preprocess the data for CNN training
    """
    print("Preprocessing data...")
    
    # Separate features and labels
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.values
    
    # Normalize pixel values (0-255 to 0-1)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to 28x28x1 for CNN
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # One-hot encode labels
    y_train = keras.utils.to_categorical(y_train, 10)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, y_train, X_test

def visualize_data(X_train, y_train):
    """
    Visualize sample digits and data distribution
    """
    print("Creating visualizations...")
    
    # Convert one-hot encoded labels back to integers for visualization
    y_train_int = np.argmax(y_train, axis=1)
    
    # Plot sample digits
    plt.figure(figsize=(12, 8))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(X_train[i].reshape(28, 28), cmap='gray')
        plt.title(f'Digit: {y_train_int[i]}')
        plt.axis('off')
    plt.suptitle('Sample Handwritten Digits from MNIST Dataset', fontsize=16)
    plt.tight_layout()
    plt.savefig('sample_digits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot digit distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_train_int)
    plt.title('Distribution of Digits in Training Set')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.savefig('digit_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def build_cnn_model():
    """
    Build the CNN model architecture
    """
    print("Building CNN model...")
    
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model Summary:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, quick=False):
    """
    Train the CNN model
    """
    print("Training the model...")
    # Quick mode: use a small subset and 1 epoch to produce results fast
    if quick:
        subset_size = min(2000, X_train.shape[0])
        print(f"Quick mode enabled: using subset of {subset_size} samples for training and 1 epoch.")
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]

        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_subset, y_subset, test_size=0.1, random_state=42, stratify=np.argmax(y_subset, axis=1)
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
        ]

        history = model.fit(
            X_train_split, y_train_split,
            batch_size=64,
            epochs=1,
            validation_data=(X_val_split, y_val_split),
            callbacks=callbacks,
            verbose=2
        )

        return history, X_val_split, y_val_split

    # Full training (original behavior)
    # Split data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=np.argmax(y_train, axis=1)
    )

    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
    ]

    # Train the model
    history = model.fit(
        X_train_split, y_train_split,
        batch_size=128,
        epochs=50,
        validation_data=(X_val_split, y_val_split),
        callbacks=callbacks,
        verbose=1
    )

    return history, X_val_split, y_val_split

def evaluate_model(model, history, X_val, y_val):
    """
    Evaluate the model and create visualizations
    """
    print("Evaluating the model...")
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Confusion Matrix
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes))
    
    return val_accuracy

def make_predictions(model, X_test):
    """
    Make predictions on test data and create submission file
    """
    print("Making predictions on test data...")
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Create submission file
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_classes) + 1),
        'Label': predicted_classes
    })
    
    submission.to_csv('submission.csv', index=False)
    print("Submission file saved as 'submission.csv'")
    
    # Show some sample predictions
    plt.figure(figsize=(12, 8))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {predicted_classes[i]}')
        plt.axis('off')
    plt.suptitle('Sample Predictions on Test Data', fontsize=16)
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return submission

def main():
    """
    Main function to run the complete pipeline
    """
    print("=" * 60)
    print("HANDWRITTEN DIGIT RECOGNITION USING MNIST DATASET")
    print("=" * 60)
    
    # parse quick/run options
    parser = argparse.ArgumentParser(description='MNIST Digit Recognition')
    parser.add_argument('--quick', action='store_true', help='Run a quick training session (1 epoch, small subset)')
    args = parser.parse_args()
    quick = args.quick

    try:
        # Load data
        train_data, test_data = load_data()
        
        # Preprocess data
        X_train, y_train, X_test = preprocess_data(train_data, test_data)
        
        # Visualize data (skip heavy visuals in quick mode)
        if not quick:
            visualize_data(X_train, y_train)
        
        # Build model
        model = build_cnn_model()
        
        # Train model (pass quick flag)
        history, X_val, y_val = train_model(model, X_train, y_train, quick=quick)
        
        # Evaluate model
        accuracy = evaluate_model(model, history, X_val, y_val)
        
        # Make predictions
        submission = make_predictions(model, X_test)
        
        print("\n" + "=" * 60)
        print("PROJECT COMPLETED SUCCESSFULLY!")
        print(f"Final Validation Accuracy: {accuracy:.4f}")
        print("Files generated:")
        print("- sample_digits.png: Sample digits visualization")
        print("- digit_distribution.png: Digit distribution plot")
        print("- training_history.png: Training curves")
        print("- confusion_matrix.png: Confusion matrix")
        print("- test_predictions.png: Sample test predictions")
        print("- submission.csv: Kaggle submission file")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure 'train.csv' and 'test.csv' files are in the current directory.")
        print("You can download them from: https://www.kaggle.com/c/digit-recognizer/data")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
