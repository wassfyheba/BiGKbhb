# -*- coding: utf-8 -*-
"""
Created on Sun Jul 27 14:20:11 2025

@author: heba_
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc, matthews_corrcoef, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (BatchNormalization, Bidirectional, GRU, 
                                   Dropout, GlobalMaxPooling1D, Dense)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


X_train=np.load('X_train_mouse_Kbhb_blosum_41.npy')
X_test=np.load('X_test_mouse_Kbhb_blosum_41.npy')
y_train=np.load("y_train_mouse_Kbhb_blosum_41.npy")
y_test=np.load("y_test_mouse_Kbhb_blosum_41.npy")

class BLOSUM62Encoder:
    """BLOSUM62 encoder for protein sequences."""
    
    def __init__(self):
        self.blosum62_matrix = {
            'A': [4,  -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],
            'R': [-1, 5,  0,  -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3],
            'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3],
            'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3],
            'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
            'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2],
            'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2],
            'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3],
            'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3],
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],
            'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2],
            'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1],
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2],
            'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2],
            'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3],
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1],
            'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],
            'X': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        }

    def encode_sequence(self, sequence):
        """Encode a single sequence."""
        encoding = []
        for aa in sequence:
            if aa in self.blosum62_matrix:
                encoding.extend(self.blosum62_matrix[aa])
            else:
                encoding.extend([0] * 20)
        return np.array(encoding)

def create_model(input_shape):
    """Create BiGRU model."""
    model = Sequential([
        BatchNormalization(input_shape=input_shape),
        Bidirectional(GRU(192, activation='relu', return_sequences=True)),
        Dropout(0.4),
        GlobalMaxPooling1D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', 
                  optimizer=Adam(learning_rate=0.001), 
                  metrics=['accuracy'])
    return model
  
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = X_train_scaled.reshape((-1, 41, 20))
X_test_scaled = X_test_scaled.reshape((-1, 41, 20))
final_model = create_model(input_shape=(41, 20))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)
mc = ModelCheckpoint('Blosum_Kbhb_BiGRU_mouse_41.keras', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
history = final_model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,  # Use 10% of training data for validation
    batch_size=64, epochs=60, verbose=1,
    callbacks=[es, mc])

y_test_pred = final_model.predict(X_test_scaled)
y_test_pred_class = (y_test_pred > 0.5).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred_class)
test_recall = recall_score(y_test, y_test_pred_class)
test_precision = precision_score(y_test, y_test_pred_class)
test_f1 = f1_score(y_test, y_test_pred_class)
test_mcc = matthews_corrcoef(y_test, y_test_pred_class)
test_auc = roc_auc_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test F1: {test_f1:.4f}")
print(f"Test MCC: {test_mcc:.4f}")
print(f"Test AUC: {test_auc:.4f}")

plt.figure(figsize=(10, 8))

fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred)
roc_auc_test = auc(fpr_test, tpr_test)
plt.plot(fpr_test, tpr_test, color='r', label=f'Test (AUC = {roc_auc_test:.3f})', lw=2, alpha=.8)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

if __name__ == "__main__":
    print("BiGKbhb Training Script")
    print("This is a template - add your data loading code here")
    
    # Example usage:
    # 1. Load your data
    # 2. Encode sequences using BLOSUM62Encoder
    # 3. Train the model using create_model function

    # 4. Evaluate and save results
