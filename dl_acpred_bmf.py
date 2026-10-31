import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Attention, GlobalAveragePooling1D
from sklearn.base import BaseEstimator, ClassifierMixin

VALID_AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(VALID_AA)}

# Qualitative properties (Hydrophobic, Polar, NegCharge, PosCharge, NeuCharge, Aromatic, Aliphatic)
# One-hot encoded to match the paper's representation
QUALC_MAP = {
    'A': [1, 0, 0, 0, 1, 0, 0], 'C': [0, 1, 0, 0, 1, 0, 0],
    'D': [0, 1, 1, 0, 0, 0, 0], 'E': [0, 1, 1, 0, 0, 0, 0],
    'F': [1, 0, 0, 0, 1, 1, 0], 'G': [1, 0, 0, 0, 1, 0, 0],
    'H': [1, 0, 0, 1, 0, 1, 0], 'I': [1, 0, 0, 0, 1, 0, 1],
    'K': [1, 0, 0, 1, 0, 0, 0], 'L': [1, 0, 0, 0, 1, 0, 1],
    'M': [1, 0, 0, 0, 1, 0, 0], 'N': [0, 1, 0, 0, 1, 0, 0],
    'P': [0, 0, 0, 0, 1, 0, 0], 'Q': [0, 1, 0, 0, 1, 0, 0],
    'R': [0, 1, 0, 1, 0, 0, 0], 'S': [0, 1, 0, 0, 1, 0, 0],
    'T': [1, 1, 0, 0, 1, 0, 0], 'V': [1, 0, 0, 0, 1, 0, 1],
    'W': [1, 0, 0, 0, 1, 1, 0], 'Y': [1, 1, 0, 0, 1, 1, 0]
}

# Quantitative Properties (Standardized Z-scores for Mass, pI, pK1, pK2, pKa, vdW)
QUANC_MAP = {
    'A': [-0.8, 0.08, 0.44, 0.69, -0.62, -0.92], 'C': [-0.01, -0.4, -1.71, 1.83, 1.34, -0.37],
    'D': [0.28, -1.48, -2.17, 0.52, -0.05, -0.44], 'E': [0.63, -1.43, -0.41, 0.65, 0.1, -0.09],
    'F': [1.08, -0.09, -2.45, -0.39, -0.62, 1.25], 'G': [-1.15, 0.05, 0.44, 0.52, -0.62, -1.45],
    'H': [0.83, 0.85, -2.51, -0.31, 0.52, 0.77], 'I': [0.23, 0.04, 0.55, 0.52, -0.62, 0.67],
    'K': [0.6, 1.76, -0.47, -0.73, 1.9, 0.75], 'L': [0.23, 0.04, 0.55, 0.52, -0.62, 0.67],
    'M': [0.68, -0.12, 0.1, -0.23, -0.62, 0.82], 'N': [0.26, -0.26, -1.37, -1.02, -0.62, 0.02],
    'P': [-0.16, 0.17, -1.54, 2.45, -0.62, -0.19], 'Q': [0.61, -0.15, -0.52, -0.39, -0.62, 0.55],
    'R': [1.3, 2.4, -0.52, -0.56, 2.37, 1.2], 'S': [-0.41, -0.12, -0.3, -0.35, -0.62, -0.87],
    'T': [-0.06, -0.17, -0.98, -0.45, -0.62, -0.16], 'V': [-0.11, 0.03, 0.33, 0.55, -0.62, 0.14],
    'W': [2.05, -0.03, 3.22, 0.11, -0.62, 3.03], 'Y': [1.47, -0.16, -0.35, -0.43, 1.79, 1.15]
}

def extract_dl_features_for_df(df, max_length=50):
    """
    Converts sequences into 3D tensors of shape (N_samples, max_length, 36)
    36 Features = 20 (BPF) + 6 (Quanc) + 10 (Qualc padded to 10)
    """
    N = len(df)
    X = np.zeros((N, max_length, 36), dtype=np.float32)
    
    for i, seq in enumerate(df["sequence"]):
        for j, aa in enumerate(seq):
            if j >= max_length:
                break
            if aa in VALID_AA:
                # 1. BPF (20 dims)
                bpf = np.zeros(20)
                bpf[AA_TO_IDX[aa]] = 1.0
                
                # 2. Quanc (6 dims)
                quanc = np.array(QUANC_MAP[aa])
                
                # 3. Qualc (10 dims - zero padded from 7)
                qualc = np.zeros(10)
                qualc[:7] = QUALC_MAP[aa]
                
                # Combine to 36 dims
                X[i, j, :] = np.concatenate([bpf, quanc, qualc])
                
    return X

def build_acpred_bmf_model(max_length=50, lstm_units=64):
    """
    Builds the Bi-LSTM + Attention architecture from ACPred-BMF.
    """
    inputs = Input(shape=(max_length, 36))
    
    # Bi-LSTM Layer
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(inputs)
    
    # Self-Attention Layer (using Keras Attention)
    attention_out = Attention()([lstm_out, lstm_out])
    
    # Pool sequences to flat vector (equivalent to weighted attention sum)
    context_vector = GlobalAveragePooling1D()(attention_out)
    
    # Dense Layers
    dense1 = Dense(50, activation='relu')(context_vector)
    
    # Output Layer
    outputs = Dense(1, activation='sigmoid')(dense1)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class KerasDLWrapper(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible wrapper for the Deep Learning Model.
    Allows seamless integration into the existing Pipeline and metric evaluation.
    """
    def __init__(self, max_length=50, lstm_units=64, epochs=35, batch_size=32):
        self.max_length = max_length
        self.lstm_units = lstm_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.model = build_acpred_bmf_model(self.max_length, self.lstm_units)
        if isinstance(X, pd.DataFrame):
            X_tensor = extract_dl_features_for_df(X, self.max_length)
        else:
            X_tensor = X
            
        self.model.fit(X_tensor, y, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        return self

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X_tensor = extract_dl_features_for_df(X, self.max_length)
        else:
            X_tensor = X
            
        preds = self.model.predict(X_tensor, verbose=0)
        # Return 2D array [prob_0, prob_1] to match sklearn API
        return np.hstack((1 - preds, preds))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)
