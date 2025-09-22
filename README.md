# BiGKbhb: Bidirectional GRU for Kbhb Site Prediction
A Bi-directional Gated Recurrent Unit Model for Predicting Lysine Beta-Hydroxybutyrylation Sites. Deep learning framework for PTM site prediction with cross-species compatibility.
A deep learning framework for predicting lysine β-hydroxybutyrylation (Kbhb) modification sites in proteins using Bidirectional Gated Recurrent Units (BiGRU) and BLOSUM62 evolutionary encoding.

# Overview
BiGKbhb is a computational tool that identifies Kbhb modification sites across multiple species with high accuracy. The model combines evolutionary-based BLOSUM62 encoding with a BiGRU neural network architecture to achieve superior performance compared to existing predictors.
Key Features

High Accuracy: Test set accuracies of 0.824 (human), 0.832 (mouse), and 0.871 (fungal)
Multi-Species Support: Pre-trained models for human, mouse, and fungal proteins
Cross-Species Prediction: General model enables prediction across diverse organisms
BLOSUM62 Encoding: Evolutionary-based sequence representation for optimal performance
Easy to Use: Simple Python interface for predictions

# Installation Prerequisites
Python 3.8 or higher
pip package manager

# Setup

Clone the repository:

bashgit clone https://github.com/wassfyheba/BiGKbhb.git
cd BiGKbhb

Install required packages:

bashpip install -r requirements.txt
Required Dependencies
numpy>=1.19.0
pandas>=1.2.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tensorflow>=2.4.0
Quick Start
Basic Usage
pythonfrom bigkbhb import BiGKbhb

Initialize the model (choose from 'human', 'mouse', 'fungal', or 'general')
model = BiGKbhb(model_type='general')

# Predict Kbhb sites for a protein sequence
sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK"
predictions = model.predict(sequence)

# Get prediction results
print(predictions)
Batch Prediction
python# Predict for multiple sequences
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK",
    "MVKVGVNGFGRIGRLVTRAAFNSGKVDVVAINDPFIDLNYMVYMFQYDSTHGK"
]

batch_predictions = model.predict_batch(sequences)
Model Architecture

# Dataset
The model was trained and evaluated on experimentally validated Kbhb sites from three species:

Human (Homo sapiens): 4,210 samples (balanced)
Mouse (Mus musculus): 1,244 samples (balanced)
Fungal (Ustilaginoidea virens): 3,104 samples (balanced)
General: Combined multi-species dataset (8,558 samples)

Datasets are available in the directory.

# Training Custom Models
To train BiGKbhb on your own data:
pythonfrom bigkbhb import BiGKbhb, BLOSUM62Encoder

# Load and encode your data
encoder = BLOSUM62Encoder()
X_train = encoder.encode_sequences(train_sequences)
y_train = train_labels

# Create and train model
model = BiGKbhb.create_model(input_shape=(41, 20))
model.fit(X_train, y_train, epochs=60, validation_split=0.1)

# Save trained model
model.save('my_kbhb_model.h5')
File Structure
BiGKbhb/
├── README.md
├── requirements.txt
├── BiGKbhb code.docx          # Original implementation code
├── data/
│   ├── human_dataset.csv
│   ├── mouse_dataset.csv
│   ├── fungal_dataset.csv
│   └── general_dataset.csv
├── models/
│   ├── human_model.keras
│   ├── mouse_model.keras
│   ├── fungal_model.keras
│   └── general_model.keras
├── src/
│   ├── __init__.py
│   ├── encoder.py             # BLOSUM62 encoding
│   ├── model.py               # BiGRU architecture
│   └── predictor.py           # Prediction interface
└── examples/
    └── prediction_example.ipynb
