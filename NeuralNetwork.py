import os
import datetime
from pyexpat import model

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

import WindowGenerator as wg

def getData():
    csv_path = 'USDRUB_210101_220101.csv'
    df = pd.read_csv(csv_path)
    date_time = pd.to_datetime(df.pop('<DATE>'), format='%d/%m/%y')
    columns = ['<OPEN>']
    df.pop('<HIGH>')
    df.pop('<CLOSE>')
    df.pop('<VOL>')
    df.pop('<LOW>')
    df.pop('<TIME>')
    return df

#Print dataframe
def printDataFrame():
    plot_features = df[columns]
    plot_features.index = date_time
    plot_features.plot(subplots=True)
    df.describe().transpose()
    plt.show()

column_indices = {name: i for i, name in enumerate(df.columns)}    
    # 70%, 20%, 10% train, validation, test
    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]

    #Normalisation dataframe
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

#Print normalized dataframe
#df_std = (df - train_mean) / train_std
#df_std = df_std.melt(var_name='Column', value_name='Normalized')
#plt.figure(figsize=(12, 6))
#ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
#ax.set_xticklabels(df.keys(), rotation=90)
#plt.show()
                       
MAX_EPOCHS = 300
val_performance = {}
performance = {}

#Function of train
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience,
                                                   mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

OUT_STEPS = 7
multi_window = wg.WindowGenerator(input_width=14, label_width=OUT_STEPS, shift=OUT_STEPS,
                                  train_df=train_df, test_df=test_df, val_df=val_df)

multi_window.plot('<OPEN>')
plt.show()

multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance = multi_lstm_model.evaluate(multi_window.val)
multi_performance = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot('<OPEN>',multi_lstm_model)
plt.show()