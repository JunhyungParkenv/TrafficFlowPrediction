import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.models import Sequential


def get_lstm(units):
    model = Sequential([
        LSTM(units[1], input_shape=(units[0], 1), return_sequences=True),
        LSTM(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='sigmoid')
    ])
    return model


def get_gru(units):
    model = Sequential([
        GRU(units[1], input_shape=(units[0], 1), return_sequences=True),
        GRU(units[2]),
        Dropout(0.2),
        Dense(units[3], activation='sigmoid')
    ])
    return model


def _get_sae(inputs, hidden, output):
    model = Sequential([
        Dense(hidden, input_dim=inputs, name='hidden', activation='sigmoid'),
        Dropout(0.2),
        Dense(output, activation='sigmoid')
    ])
    return model


def get_saes(layers):
    models = [Sequential() for _ in range(4)]
    models[0] = _get_sae(layers[0], layers[1], layers[-1])
    models[1] = _get_sae(layers[1], layers[2], layers[-1])
    models[2] = _get_sae(layers[2], layers[3], layers[-1])

    models[3] = Sequential([
        Dense(layers[1], input_dim=layers[0], name='hidden1', activation='sigmoid'),
        Dense(layers[2], name='hidden2', activation='sigmoid'),
        Dense(layers[3], name='hidden3', activation='sigmoid'),
        Dropout(0.2),
        Dense(layers[4], activation='sigmoid')
    ])

    return models
