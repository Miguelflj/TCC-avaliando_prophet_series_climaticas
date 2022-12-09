
from re import X
from numpy import NaN, dsplit, empty, float64
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta,date

import numpy as np
import tensorflow as tf
from IPython.display import clear_output
import keras
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Dropout

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error

import seaborn as sns
import sys

from connect import DATABASE

def load_dataset(estacao):
    df = pd.DataFrame(db.get_data_of_estations(estacao), columns=['Codigo','Data','Precipitacao','Temperatura','Umidade','Imputado'])
    df = df.astype({'Precipitacao':float64, 'Temperatura': float64, 'Umidade':float64})
    
    return df



def train_model(model, X_train, Y_train, validation, callbacks):

    model.fit(X_train, Y_train, epochs=50, batch_size=128, validation_data=validation, callbacks=callbacks)
    return model

def split_train_test(df,split_size):
    train_size = int(len(df) * split_size)
    train, val = df.iloc[0:train_size], df.iloc[train_size:]
    dev_size = int(len(val) * 0.5)
    dev = val.iloc[0:dev_size]
    return train,dev,val

def z_score(df):
    df_std = df.copy()
    properties = []
    for column in ['Precipitacao','Umidade','Temperatura']:
        properties.append([df_std[column].mean(), df_std[column].std()])
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std,properties


def show_decompose(df):
    new_df = df[['Data','Precipitacao']]
    new_df.set_index('Data',inplace=True)
    new_df.index=pd.to_datetime(new_df.index)
    print(new_df)
    result = seasonal_decompose(new_df, period=365)
    result.plot()
    plt.show()

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_plot(plot_data, delta, title):
    labels = ["History", "True Future", "Model Prediction"]
    marker = [".-", "rx", "go"]
    time_steps = list(range(-(plot_data[0].shape[0]), 0))
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, val in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel("Time-Step")
    plt.show()
    return

def cut_dataframes(df,date_start_fhalf,date_end_fhalf,date_shalf):
        
        df_train1 = df.loc[ ( ( df['Data'] >= date(date_start_fhalf[0],date_start_fhalf[1],date_start_fhalf[2])) & ( df['Data'] < date(date_end_fhalf[0],date_end_fhalf[1],date_end_fhalf[2]) ))] 
        df_train2 = df.loc[(df['Data'] >= date(date_shalf[0],date_shalf[1],date_shalf[2]))] 
        frames =  [df_train1,df_train2]
        df_train = pd.concat(frames)
        return df_train


if __name__ == "__main__":
    
    db = DATABASE(
        "estacoes",
        "postgres",
        "123"
        )

    estations = db.get_estations()
    df_estacoes = pd.DataFrame(estations,columns=['Codigo','Nome',"Sigla","Longitude","Latitude", "Altitude"])
    
    if(len(sys.argv) >= 3):

        for index,row in df_estacoes.iterrows():
        
            if(row[0] != 83995):
        
                
                
                df = load_dataset(row[0])

                df = df.sort_values(by='Data')
                df_std,props = z_score(df)
                
                #df_corr = df[['Precipitacao','Umidade','Temperatura']]
                #correlation = df_corr.corr()

                #plot = sns.heatmap(correlation, annot = True, fmt=".1f", linewidths=.6)
                #plt.title(row[1])
                #plt.show()
                
                

                epochs = int(sys.argv[1])
                case = int(sys.argv[2])
                weather = str(sys.argv[3])
                
                if(weather == 'Temperatura'):
                    features = ['Umidade','Precipitacao']
                elif(weather == 'Precipitacao'):
                    features = ['Umidade','Temperatura']
                elif(weather == 'Umidade'):
                    features = ['Temperatura','Precipitacao']
                else:
                    print("Error: feature not found")
                if(case == 1):
                    df_train = df_std.loc[( df_std['Data'] < date(2018,1,1))]
                    df_test = df_std.loc[ df_std['Data'] >= date(2018,1,1)]
                elif(case == 2):
                    df_train = cut_dataframes(df_std,[2001,1,1],[2010,3,21],[2010,9,23])
                    df_test = df_std.loc[( df_std['Data'] >= date(2010,3,21)) & (df_std['Data'] < date(2010,9,23))]
                elif(case == 3):
                    #caso 03: verao e primavera
                    df_train = cut_dataframes(df_std,[2001,1,1],[2010,9,23],[2011,3,21])
                    df_test = df_std.loc[( df_std['Data'] >= date(2010,9,23)) & (df_std['Data'] < date(2011,3,21))]
                elif(case == 4):
                    #caso 04: inverno e primavera
                    df_train = cut_dataframes(df_std,[2001,1,1],[2010,6,21],[2010,12,21])
                    df_test = df_std.loc[( df_std['Data'] >= date(2010,6,21)) & (df_std['Data'] < date(2010,12,21))]
                elif(case == 5):
                    #caso05: verao e outono
                    df_train = cut_dataframes(df_std,[2001,1,1],[2010,12,21],[2011,6,21])
                    df_test = df_std.loc[( df_std['Data'] >= date(2010,12,21)) & (df_std['Data'] < date(2011,6,21))]
                else:
                    print("Case test don't found!")    
                

                
                df_train.index = df_train['Data']
                df_test.index = df_test['Data']

                df_train.drop('Data',axis=1,inplace=True)
                df_test.drop('Data',axis=1,inplace=True)

                

                train_data = df_train 
                val_data = df_test
                
                

                step = 1
                past = 7
                future = 1
                learning_rate = 0.001
                batch_size = 512
                #epochs = 8

                

                

                x_train = train_data[features]
                
                y_train = train_data[[weather]]
                
                print(x_train.head(10))

                
                
                
                dataset_train = tf.keras.utils.timeseries_dataset_from_array(
                    x_train,    
                    y_train,
                    sequence_length=1,
                    sampling_rate=step,
                    batch_size=1,
                )

                
                x_val = val_data[features]
                
                y_val = val_data[[weather]]
                


                
                

                
                dataset_val = tf.keras.utils.timeseries_dataset_from_array(
                    x_val,
                    y_val,
                    sequence_length=1,
                    sampling_rate=step,
                    batch_size=1,
                )
                print(dataset_val)

                    
                for batch in dataset_train.take(1):
                    inputs, targets = batch


                

                
                #Modelo
                inputs = keras.layers.Input(shape=(inputs.shape[1], inputs.shape[2]))
                lstm_out = keras.layers.LSTM(32)(inputs)
                outputs = keras.layers.Dense(1)(lstm_out)

                model = keras.Model(inputs=inputs, outputs=outputs)
                model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
                model.summary()
                
                path_checkpoint = "model_checkpoint.h5"
                es_callback = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=5)

                modelckpt_callback = keras.callbacks.ModelCheckpoint(
                    monitor="val_loss",
                    filepath=path_checkpoint,
                    verbose=1,
                    save_weights_only=True,
                    save_best_only=True,
                )
                
                
                history = model.fit(
                    dataset_train,
                    epochs=epochs,
                    validation_data=dataset_val,
                    callbacks=[es_callback, modelckpt_callback],
                    verbose=False
                )

                
                
                model.save('myModel.h5')
                #visualize_loss(history, "Training and Validation Loss")


                y_pred = []
                for x,y in dataset_val:
                    #print(x)
                    #print(y)
                    value = model.predict(x, verbose=False)
                    y_pred.append(value[0][0])
                
                
                
                dataset_pred = pd.DataFrame()
                dataset_pred['Data'] = df_test.index
                dataset_pred['y_pred'] = y_pred[:]
                dataset_pred['y_true'] = y_val[weather].values
                
                print(dataset_pred.head())
                if( weather == 'Precipitacao'):
                    mean,std = props[0]
                elif( weather == 'Umidade'):
                    mean,std = props[1]
                elif( weather == 'Temperatura'):
                    mean,std = props[2]
                else:
                    print("Fail, weather not found")
                    break

                dataset_pred['y_pred'] = (dataset_pred['y_pred'] * std) + mean
                dataset_pred['y_true'] = (dataset_pred['y_true'] * std) + mean
                dataset_pred.to_csv('csv_output/'+weather.capitalize()+'_caso'+str(case)+'_'+str(row[0])+'_LSTM.csv',index=False)
                
                
                print("Estação:{}\nMAE:{:.3f}\nMAPE:{:.3f}\nMSE:{:.3f}\nRMSE:{:.3f}".format(
                    row[1],
                    mean_absolute_error(y_true=dataset_pred['y_true'],y_pred=dataset_pred['y_pred']),
                    mean_absolute_percentage_error(y_true=dataset_pred['y_true'],y_pred=dataset_pred['y_pred']),
                    mean_squared_error(y_true=dataset_pred['y_true'], y_pred=dataset_pred['y_pred']),
                    mean_squared_error(y_true=dataset_pred['y_true'], y_pred=dataset_pred['y_pred'],squared=False)
                    ))

                
                #dataset_pred['y_true'] = val_data['Precipitacao']
                
                
                
                for x, y in dataset_val.take(5):
                    
                    show_plot(
                        [x[0][:, 1].numpy(), y[0].numpy(), model.predict(x)[0]],
                        12,
                        "Single Step Prediction",
                    )
                
    else:
        print("Necessary arguments: python3 analisys_data.py <number of epochs> <testes,[1,5]> <variavel,[Temperatura, Umidade, Precipitacao] ")
        print("#Teste 1 sem Regressors - Ex.: python3 analisys_data.py 10 1 Temperatura")           