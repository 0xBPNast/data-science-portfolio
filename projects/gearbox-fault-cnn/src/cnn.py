"""
cnn.py — 1D Convolutional Neural Network for Fault Classification

Implements the 1D-CNN architecture from Jing et al. (2017), adapted for
3-class gearbox fault classification:
    0 — Healthy
    1 — Cracked Tooth
    2 — Worn Teeth

Architecture
------------
    Input        : (segment_length, 1)
    Conv1D       : 10 filters, kernel size 64, ReLU
    MaxPooling1D : pool size 2
    Flatten      : —
    Dense        : 30 nodes, ReLU
    Dense        : 3 nodes, Softmax

References
----------
Jing, L. et al. (2017). An adaptive multi-sensor data fusion method based
    on deep convolutional neural networks for fault diagnosis of
    planetary gearbox.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


# ── Model Definition ───────────────────────────────────────────────────────────

def build_1d_cnn(input_length, num_classes=3):
    """
    1D-CNN architecture from Jing et al. (2017).

    Single convolution + pooling block followed by dense classification.
    ReLU activation mitigates vanishing gradients and enforces sparsity.
    Softmax output produces class probability distribution.

    Parameters
    ----------
    input_length : int — segment length (CNN input size)
    num_classes  : int — number of output classes (default 3)

    Returns
    -------
    model : compiled Keras Sequential model
    """
    model = Sequential([
        Input(shape=(input_length, 1)),
        Conv1D(filters=10, kernel_size=64, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(30, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=SGD(learning_rate=0.02, momentum=0.5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ── Normalisation ──────────────────────────────────────────────────────────────

def normalize_and_standardize(X_train, X_val, X_test):
    """
    MinMax scaling followed by StandardScaler normalisation.

    Scalers are fit on training data only and applied to validation
    and test sets to prevent data leakage.

    Parameters
    ----------
    X_train, X_val, X_test : arrays — shape (n, segment_length)

    Returns
    -------
    X_train_n, X_val_n, X_test_n : arrays — shape (n, segment_length, 1)
                                             ready for 1D-CNN input
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_tr_s = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
    X_va_s = scaler.transform(X_val.reshape(X_val.shape[0], -1))
    X_te_s = scaler.transform(X_test.reshape(X_test.shape[0], -1))

    standardizer = StandardScaler()
    X_tr_n = standardizer.fit_transform(X_tr_s).reshape(X_train.shape[0], X_train.shape[1], 1)
    X_va_n = standardizer.transform(X_va_s).reshape(X_val.shape[0],   X_val.shape[1],   1)
    X_te_n = standardizer.transform(X_te_s).reshape(X_test.shape[0],  X_test.shape[1],  1)

    return X_tr_n, X_va_n, X_te_n


def normalize_for_inference(X, fit_data=None):
    """
    Apply MinMax + StandardScaler for single-dataset inference.

    If fit_data is provided, scalers are fit on fit_data (use training
    data to avoid leakage). Otherwise fit on X itself.

    Parameters
    ----------
    X        : array — shape (n, segment_length)
    fit_data : array — optional data to fit scalers on

    Returns
    -------
    X_norm : array — shape (n, segment_length, 1)
    """
    fit = fit_data if fit_data is not None else X

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(fit.reshape(fit.shape[0], -1))
    X_scaled = scaler.transform(X.reshape(X.shape[0], -1))

    standardizer = StandardScaler()
    standardizer.fit(scaler.transform(fit.reshape(fit.shape[0], -1)))
    X_norm = standardizer.transform(X_scaled).reshape(X.shape[0], X.shape[1], 1)

    return X_norm


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train_and_evaluate_1d_cnn(X_train, y_train, X_val, y_val, X_test, y_test,
                               num_classes=3, epochs=100, batch_size=50,
                               save_path=None):
    """
    Train, evaluate and optionally save a 1D-CNN model.

    Parameters
    ----------
    X_train/val/test : arrays — shape (n, segment_length, 1)
    y_train/val/test : arrays — one-hot encoded labels (n, num_classes)
    num_classes      : int   — number of output classes
    epochs           : int   — training epochs
    batch_size       : int   — SGD mini-batch size
    save_path        : str   — save model to this path (None = don't save)
                               Extension '.keras' appended automatically.

    Returns
    -------
    history       : Keras History object
    test_accuracy : float — test set classification accuracy
    cm            : array — confusion matrix (num_classes x num_classes)
    """
    model = build_1d_cnn(X_train.shape[1], num_classes)

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    if save_path:
        model.save(f'{save_path}.keras')

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm     = confusion_matrix(y_true, y_pred)

    return history, test_accuracy, cm
