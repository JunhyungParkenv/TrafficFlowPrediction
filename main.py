import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data
import tensorflow as tf
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    sums = sum(abs(y - yp) / y for y, yp in zip(y, y_pred))
    mape = (sums / len(y_pred)) * 100
    return mape

def eva_regress(y_true, y_pred):
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

def plot_results(y_true, y_preds, names):
    d = '2016-3-4 00:00'
    x = pd.date_range(d, periods=288, freq='5min')
    fig, ax = plt.subplots()
    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)
    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    fig.autofmt_xdate()
    plt.show()

def main():
    lstm = tf.keras.models.load_model('model/lstm.h5')
    gru = tf.keras.models.load_model('model/gru.h5')
    saes = tf.keras.models.load_model('model/saes.h5')
    models = [lstm, gru, saes]
    names = ['LSTM', 'GRU', 'SAEs']
    lag = 12
    file1 = 'data/train.csv'
    file2 = 'data/test.csv'
    _, _, X_test, y_test, scaler = process_data(file1, file2, lag)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_preds = []
    for name, model in zip(names, models):
        X_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1) if name != 'SAEs' else (X_test.shape[0], X_test.shape[1]))
        predicted = model.predict(X_reshaped)
        predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).flatten()
        y_preds.append(predicted[:288])
        print(name)
        eva_regress(y_test[:288], predicted[:288])
    plot_results(y_test[:288], y_preds, names)

if __name__ == '__main__':
    main()
