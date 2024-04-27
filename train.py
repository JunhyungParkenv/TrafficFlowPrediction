import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
import tensorflow as tf
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save(f'model/{name}.h5')
    pd.DataFrame.from_dict(hist.history).to_csv(f'model/{name}_loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config):
    temp = X_train

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = tf.keras.Model(inputs=p.input, outputs=p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        m.fit(temp, y_train, batch_size=config["batch"], epochs=config["epochs"], validation_split=0.05)
        models[i] = m

    saes = models[-1]
    for i, m in enumerate(models[:-1]):
        weights = m.get_layer('hidden').get_weights()
        saes.get_layer(f'hidden{i+1}').set_weights(weights)

    train_model(saes, X_train, y_train, name, config)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", help="Model to train.")
    args = parser.parse_args(argv[1:])

    config = {"batch": 256, "epochs": 600}
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    X_train, y_train, _, _, _ = process_data(file1, file2, 12)

    if args.model == 'lstm':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_lstm([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        m = model.get_gru([12, 64, 64, 1])
        train_model(m, X_train, y_train, args.model, config)
    elif args.model == 'saes':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        m = model.get_saes([12, 400, 400, 400, 1])
        train_seas(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
