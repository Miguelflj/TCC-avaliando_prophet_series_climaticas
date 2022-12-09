import pandas as pd
from connect import DATABASE
from numpy import NaN, dsplit, empty, float64
from keras.models import load_model
import tensorflow as tf
from datetime import timedelta,date


def load_dataset():
    df = pd.DataFrame(db.get_data_of_estations(row[0]), columns=['Codigo','Data','Precipitacao','Temperatura','Umidade','Imputado'])
    df = df.astype({'Precipitacao':float64, 'Temperatura': float64, 'Umidade':float64})
    #df['Data'] = pd.to_datetime(df['Data'])

    return df



def z_score(df):
    df_std = df.copy()
    properties = []
    for column in ['Precipitacao','Umidade','Temperatura']:
        properties.append([df_std[column].mean(), df_std[column].std()])
        #print("Mean:{} and Std:{}".format(df_std[column].mean(),df_std[column].std()))
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
    return df_std,properties

def z_score_to_default(df_std, properties):
    df_default = df_std.copy()
    index = 0
    for column in ['Precipitacao','Umidade','Temperatura']:
        mean, std = properties[index]
        df_default[column] = (df_default[column] * std )+ mean
        index += 1

    return df_default

def cut_dataframes(df,date_start_fhalf,date_end_fhalf,date_shalf):
        
        df_train1 = df.loc[ ( ( df['ds'] >= date(date_start_fhalf[0],date_start_fhalf[1],date_start_fhalf[2])) & ( df['ds'] < date(date_end_fhalf[0],date_end_fhalf[1],date_end_fhalf[2]) ))] 
        df_train2 = df.loc[(df['ds'] >= date(date_shalf[0],date_shalf[1],date_shalf[2]))] 
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
    for index,row in df_estacoes.iterrows():
        
        if(row[0] == 83587):
            df = load_dataset()
            model = load_model('/home/miguel/Documents/TCC/scripts/myModel.h5')
            case = 1

            df = df[['Data','Precipitacao','Umidade','Temperatura']]
            if case == 1:
                df_std,props =  z_score(df)
                df_train = df_std.loc[( df_std['Data'] < date(2018,1,1))]
                df_test = df_std.loc[ df_std['Data'] >= date(2018,1,1)]

                df_train.index = df_train['Data']
                df_test.index = df_test['Data']

                df_train.drop('Data',axis=1,inplace=True)
                df_test.drop('Data',axis=1,inplace=True)

            elif case == 2:
                df_std,props =  z_score(df)
                df_train = cut_dataframes(df_std,[2001,1,1],[2010,3,21],[2010,9,23])
                df_test = df_std.loc[( df_std['Data'] >= date(2010,3,21)) & (df_std['Data'] < date(2010,9,23))]

                df_train.index = df_train['Data']
                df_test.index = df_test['Data']          
           

            print(df_train.head())
            print(df_test.head())
        
            
            
            
            dataset_val = tf.keras.utils.timeseries_dataset_from_array(
                df_train,
                df_test,
                sequence_length=1,
                sampling_rate=1,
                batch_size=512,

            )
           
            for x,y in dataset_val:

                output_predict = model.predict(x)
                