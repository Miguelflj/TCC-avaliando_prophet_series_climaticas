
from datetime import timedelta,date
from os import listdir, mkdir,rename
from os.path import isfile, join
import time
from warnings import catch_warnings
from numpy import NaN, empty, float64
import pandas as pd
import numpy as np
import math
from prophet import Prophet
from prophet.diagnostics import cross_validation,performance_metrics
from prophet.plot import plot_cross_validation_metric
import numpy as np
from connect import DATABASE
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, r2_score
import sys




class DATA_MANIPULATION:

    def __init__(self, dir, sgs, total_data, limit_perc):
        self.dir = dir
        
        self.SGS = sgs
        # Cerca de 20 anos de dados climáticos diários.
        self.TOTAL_DATA = total_data
        self.LIMIT_MISSING_DATA_PERC = limit_perc #porcentagem
        self.TO_DISCARD = [] 
        self.EMPTY = []
        self.BETTER_STATIONS = []
        self.MODEL_DATA = []
        self.LIMIT_TO_BETTER = 1
        self.TOTAL_STATIONS_PROCESS = 0

    
    def analisys_data(self):
        file_analisys = open(self.dir+"Analise_estacoes.txt",'w+')
        for sg in self.SGS:
            files = [f for f in listdir(self.dir+sg) if isfile(join(self.dir+sg,f))]
            for f in files:
                self.TOTAL_STATIONS_PROCESS += 1
                discard_curr = False
                empty_curr = False
                better = False
                s_model = False
                df = pd.read_csv(self.dir + sg+ '/' +f )
             
                self.get_biggestSequence_missingData(df, f,file_analisys)
                name_station = f.replace('.csv', '')
                print("Estação:{}".format(name_station))
                #Quantidade de linhas 7670
                count_nulls = df.isnull().sum(axis = 0)
                print("Quantidade de dados faltantes.")
                
                prep_porcentagem = ((count_nulls[1]*100) / self.TOTAL_DATA)
                temp_porcentagem = ((count_nulls[2]*100) / self.TOTAL_DATA)
                umid_porcentagem = ((count_nulls[3]*100) / self.TOTAL_DATA)
               
                #verifica se uma das colunas de medição estão vazias.
                if (( count_nulls[1] == self.TOTAL_DATA ) or ( count_nulls[2] == self.TOTAL_DATA ) or (count_nulls[3] == self.TOTAL_DATA)):
                    empty_curr = True
                
                #Primeiro teste verifica se estao todas as datas.
                if( ( prep_porcentagem >= self.LIMIT_MISSING_DATA_PERC )  or ( temp_porcentagem > self.LIMIT_MISSING_DATA_PERC) or (umid_porcentagem > self.LIMIT_MISSING_DATA_PERC) ):
                    discard_curr = True
                else:
                    better = True

                #series modelos.
                if( (count_nulls[1] < 10) and (count_nulls[2] < 10) and (count_nulls[1] < 10)):
                    s_model = True
                print("Prep.: {}, Porcentagem: {:.2f}%".format(count_nulls[1],prep_porcentagem)) 
                print("Temp.: {}, Porcentagem: {:.2f}%".format(count_nulls[2],temp_porcentagem))
                print("Umid.: {}, Porcentagem: {:.2f}%".format(count_nulls[3],umid_porcentagem))
                print("")
                if(empty_curr):
                    self.EMPTY.append(name_station)
                if(discard_curr and not empty_curr):    
                    self.TO_DISCARD.append(name_station)
                if(better):
                    self.BETTER_STATIONS.append(name_station)
                if(s_model):
                    self.MODEL_DATA.append(name_station)

        file_analisys.close()
    
    
    def process_df(self):
        for sg in self.SGS:
            files = [f for f in listdir(self.dir+sg) if isfile(join(self.dir+sg,f))]
            for f in files:
                df = pd.read_csv(self.dir + sg+ '/' +f )
                
               
          
    def get_biggestSequence_missingData(self, df, name, file_analisys):
        biggest_sequence_prep = 0
        biggest_sequence_temp = 0
        biggest_sequence_umid = 0
        count_prep = 0
        count_temp = 0
        count_umid = 0
        miss_prep = False
        miss_temp = False
        miss_umid = False
        for index, row in df.iterrows():
            if(pd.isna(row[1])):
                
                if(miss_prep):
                    
                    count_prep += 1
                else:
                    miss_prep = True
                    count_prep = 1
            else:
                if( biggest_sequence_prep  < count_prep ):
                        biggest_sequence_prep = count_prep
                count_prep = 0
                miss_prep = False
            if(pd.isna(row[2])):
                if(miss_temp):
                    count_temp += 1
                else:
                    miss_temp = True
                    
                    count_temp = 1
            else:
                if( biggest_sequence_temp < count_temp ):
                        biggest_sequence_temp = count_temp
                count_temp = 0
                miss_temp = False
            if(pd.isna(row[3])):
                if(miss_umid):
                    count_umid += 1
                else:
                    miss_umid = True
                    
                    
                    count_umid = 1
            else:
                if( biggest_sequence_umid < count_umid ):
                        biggest_sequence_umid = count_umid
                count_umid = 0
                miss_umid = False
        file_analisys.write("Name: {}\n".format(name))
        file_analisys.write("Biggest sequence of Missing data:Prep.:{} Temp.:{} Umid.:{}\n".format(biggest_sequence_prep,biggest_sequence_temp,biggest_sequence_umid))


    def record_output(self,file, name_file):
        with open(self.dir+name_file, 'w+') as f:
            for stt in file:
                f.write(stt + '\n')
    
    
    def load_banco_estacoes(self, db):
        try:
            for sg in self.SGS:  
                file_estacoes = open(dir + 'headers_info_'+sg+'.txt', 'r')
                data_estado = file_estacoes.readlines()
                for i in range(10, len(data_estado)):
                    estacao = data_estado[:i]
                    nome = estacao.split()[-1]
                    cod = estacao.split()[-1]
                    lat = estacao.split()[-1]
                    long = estacao.split()[-1]
                    alt = estacao.split()[-1]
                    print("Insere: Cod:{} Nome:{} SG:{} Lat:{} Long:{} Alt:{}".format(cod,nome,sg,long,lat,alt))
                    #adiciona no banco db.insert_estacoes(cod,nome,sg,long,latl,alt)

        except Exception as e:
            print('error', e)


    def load_banco_dadosDia(self,db):
        try:
            for sg in self.SGS:
                files = [f for f in listdir(self.dir+sg) if isfile(join(self.dir+sg,f))]
                for f in files:
                    df = pd.read_csv(self.dir + sg+ '/' +f )
                    for index,row in df.iterrows():
                        print("Data:{} Precipitacao:{} Temperatura:{} Umidade{}".format(row[0],row[1],row[2],row[3]))

        except Exception as e:
            print('error', e)

    def get_df_of_files(self,db, file_name):
        file = open(self.dir + file_name, 'r')
        content = file.readlines()
        for line in content:
            state = line.split('_')[-1].replace('\n','')
            name_df = line.replace('\n','')+'.csv'
            name_file = line.replace('\n','')+'.txt'
            df = pd.read_csv(self.dir+state+'/'+name_df)
            header = open(self.dir+state+'/header/'+name_file,'r')
            content = header.readlines()
            name = content[0].replace('Nome:','').strip().replace('\n','')
            cod = content[1].split()[2]
            lat = content[2].split()[1]
            long = content[3].split()[1]
            alt = content[4].split()[1]
            db.insert_estacoes(cod,name,state,lat,long,alt)
            for index,row in df.iterrows():
                
                data = row[0]
                prep = row[1]
                temp = row[2]
                umid = row[3]
                imput = row[4]
                new_data = str(data)
               
                if(math.isnan(prep) or math.isnan(temp) or math.isnan(umid) ):
                    if(math.isnan(prep)):
                        prep = 'NULL'
                    if(math.isnan(temp)):
                        temp = 'NULL'
                    if(math.isnan(umid)):
                        umid = 'NULL'
                    
                
                db.insert_dados_diarios(cod,new_data,prep,temp,umid,imput)
            
    def imput_nulls_with_media(self, df, df_nulls):
        #print(df.head())
        #print(df_nulls.head())
        
        for index,row in df_nulls.iterrows():
            
            count_prep = 0
            media_prep = 0
            count_temp = 0
            media_temp = 0
            count_umid = 0
            media_umid = 0
            
            for i in range(3,0,-1):
                
                
                value = df.loc[(row[1] - timedelta(i))]['Precipitacao']
                if( not math.isnan(value) ):
                    #print("Prep - Data: {} Valor anterior: {}".format(row[1] - timedelta(i),value))
                    count_prep += 1
                    media_prep += value
                value = df.loc[(row[1] + timedelta(i))]['Precipitacao']
                if( not math.isnan(value) ):
                    #print("Prep - Data: {} Valor posterior: {}".format(row[1] + timedelta(i),value))
                    count_prep += 1
                    media_prep += value
                
                
                value =  df.loc[(row[1] - timedelta(i))]['Temperatura']
                if( not math.isnan(value) ):
                   # print("Temp - Data: {} Valor anterior: {}".format(row[1] - timedelta(i),value))
                    count_temp += 1
                    media_temp += value
                value = df.loc[(row[1] + timedelta(i))]['Temperatura']
                if( not math.isnan(value) ):
                    #print("Temp - Data: {} Valor posterior: {}".format(row[1] + timedelta(i),value))
                    count_temp += 1
                    media_temp += value

                value =  df.loc[(row[1] - timedelta(i))]['Umidade']
                if( not math.isnan(value) ):
                    #print("Umid - Data: {} Valor anterior: {}".format(row[1] - timedelta(i),value))
                    count_umid += 1
                    media_umid += value
                value = df.loc[(row[1] + timedelta(i))]['Umidade']
                if( not math.isnan(value) ):
                    #print("Umid - Data: {} Valor posterior: {}".format(row[1] + timedelta(i),value))
                    count_umid += 1
                    media_umid += value
            
            if(math.isnan(df.loc[row[1],'Precipitacao'])):
                df.loc[row[1],'Precipitacao'] = media_prep/count_prep
                df.loc[row[1],'Imputado'] = 1
            if(math.isnan(df.loc[row[1], 'Temperatura'])):
                df.loc[row[1],'Temperatura'] = media_temp/count_temp
                df.loc[row[1],'Imputado'] = 1
            if(math.isnan(df.loc[row[1],'Umidade'])):
                df.loc[row[1],'Umidade'] = media_umid/media_umid
                df.loc[row[1],'Imputado'] = 1

    def set_imputados(self,db, df, df_nulls):
        
        for index,row in df_nulls.iterrows():
           
            if(df.loc[row[1],'Codigo'] == row[0] ):
                #data, cod,prep,temp,umid,imput
                db.set_imput_data(row[1], row[0], df.loc[row[1],'Precipitacao'], df.loc[row[1], 'Temperatura'], df.loc[row[1], 'Umidade'], df.loc[row[1],'Imputado'])

    
def cut_dataframes(df,date_start_fhalf,date_end_fhalf,date_shalf):
        
        df_train1 = df.loc[ ( ( df['ds'] >= date(date_start_fhalf[0],date_start_fhalf[1],date_start_fhalf[2])) & ( df['ds'] < date(date_end_fhalf[0],date_end_fhalf[1],date_end_fhalf[2]) ))] 
        df_train2 = df.loc[(df['ds'] >= date(date_shalf[0],date_shalf[1],date_shalf[2]))] 
        frames =  [df_train1,df_train2]
        df_train = pd.concat(frames)
        return df_train

def accuracy(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    abs_error = np.abs(y_true - y_pred)
    fa = 1 - (abs_error / y_true)
    total_y = np.sum(y_true)
    total_error = np.sum(abs_error)
    return (1 - (total_error / total_y))



if __name__=='__main__':
    #sgs = ['AC','AL','AM','AP','BA','CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA','PB','PE', 'PI', 'PR', 'RJ','RN','RO','RR','RS','SC','SE','SP','TO']
    data = DATA_MANIPULATION(
        '/home/miguel/Documents/TCC/Estados/',
        
        ['AC','AL','AM','AP','BA','CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA','PB','PE', 'PI', 'PR', 'RJ','RN','RO','RR','RS','SC','SE','SP','TO'],
        7670,
        30
    )
    db = DATABASE(
        "estacoes",
        "postgres",
        "123"
        )
    #data.get_df_of_files(db, 'Estacoes_modelo.txt')
     
    #outono 21 de março a 21 de junho
    #Inverno 21 de junho a 23 de setembro
    #primavera 23 de setembro a 21 de dezembro
    #verao 21 de dezembro a 21 de março



    estations = db.get_estations()
    df_estacoes = pd.DataFrame(estations,columns=['Codigo','Nome',"Sigla","Longitude","Latitude", "Altitude"])

    if(len(sys.argv) >= 3):
    
        
        output_file = open("Prophet-OutPut.txt",'w+')
        for index,row in df_estacoes.iterrows():
            
            if(row[0] != 83995):

                df_dados = pd.DataFrame(db.get_data_of_estations(row[0]), columns=['Codigo','Data','Precipitacao','Temperatura','Umidade','Imputado'])
                df_dados = df_dados.astype({'Precipitacao':float64, 'Temperatura': float64, 'Umidade':float64})
                

                
                    
                print("Primeiro argumento: {}".format(sys.argv[1]))
                    
                df_dados = df_dados.sort_values(by='Data')
                regressors = int(sys.argv[1])
                weather = str(sys.argv[3])
                    
                if(regressors):
                    print("Wheather:{}".format(weather))
                    if( weather == 'Temperatura'):
                        
                        add1 = 'Umidade'
                        add2 = 'Precipitacao'
                    elif(weather == 'Umidade'):
                        add1 = 'Precipitacao'
                        add2 = 'Temperatura'
                    elif(weather == 'Precipitacao'):
                        
                        add1 = 'Temperatura'
                        add2 = 'Umidade'
                    else:
                        print("Variável não encontrada, verifique: {}".format(weather))
                        break

                    df = df_dados[['Data', weather,add1,add2]]
                    df.columns = ['ds','y','add1','add2']
                else:
                    
                    
                    if(weather in ['Temperatura', 'Precipitacao','Umidade']):
                        df = df_dados[['Data', weather]]
                        df.columns = ['ds','y']
                    else:
                        print("Variável não encontrada, verifique: {}".format(weather))
                        break
                
                print("Segundo argumento: {}".format(sys.argv[2]))
                caso = int(sys.argv[2])
                if(caso == 1):
                    #caso 01:
                    df_train = df.loc[( df['ds'] < date(2018,1,1))]
                    df_test = df.loc[ df['ds'] >= date(2018,1,1)]
                elif(caso == 2):    
                    #caso02: outono e inverno
                    df_train = cut_dataframes(df,[2001,1,1],[2010,3,21],[2010,9,23])
                    df_test = df.loc[( df['ds'] >= date(2010,3,21)) & (df['ds'] < date(2010,9,23))]
                elif(caso == 3):
                    #caso 03: verao e primavera
                    df_train = cut_dataframes(df,[2001,1,1],[2010,9,23],[2011,3,21])
                    df_test = df.loc[( df['ds'] >= date(2010,9,23)) & (df['ds'] < date(2011,3,21))]
                elif(caso == 4):
                    #caso 04: inverno e primavera
                    df_train = cut_dataframes(df,[2001,1,1],[2010,6,21],[2010,12,21])
                    df_test = df.loc[( df['ds'] >= date(2010,6,21)) & (df['ds'] < date(2010,12,21))]
                elif(caso == 5):
                    #caso05: verao e outono
                    df_train = cut_dataframes(df,[2001,1,1],[2010,12,21],[2011,6,21])
                    df_test = df.loc[( df['ds'] >= date(2010,12,21)) & (df['ds'] < date(2011,6,21))]
                elif(caso == 6):
                    df_train = df
                else:
                    print("Caso de teste não existe.")
                
                m = Prophet(daily_seasonality=False,
                            weekly_seasonality=False,
                            yearly_seasonality=True,
                            
                            )
                if(regressors):
                    m.add_regressor('add1')    
                    m.add_regressor('add2')
                
                m.fit(df_train)

                #future = m.make_future_dataframe(periods=365)
                #future = m.make_future_dataframe(periods=df_test.shape[0])
                if(caso == 6):
               
                    df_cv = cross_validation(m, initial ="1460 days", period="547 days" ,horizon= "180 days")
                    df_p = performance_metrics(df_cv, rolling_window=2)
                    print(df_cv.describe)
                    print(df_p.head())
                    try:
                        mkdir("/home/miguel/Documents/TCC/scripts/cross_visualizations")
                    except Exception as e:
                        pass
                    fig4 = plot_cross_validation_metric(df_cv, metric='rmse')
                    fig4.savefig("cross_visualizations/"+str(row[0])+'_mse.png')
                    fig5 = plot_cross_validation_metric(df_cv, metric='mae')
                    fig5.savefig("cross_visualizations/"+str(row[0])+'_mae.png')
                    fig6 = plot_cross_validation_metric(df_cv, metric='coverage')
                    fig6.savefig("cross_visualizations/"+str(row[0])+'_coverage.png')
                else:
                    forecast = m.predict(df_test.drop(columns=['y']))

                    
                    dataframe_output = pd.DataFrame()
                    dataframe_output['data'] = forecast['ds'].values
                    print(forecast.head()[['ds','yhat_lower','yhat_upper','yhat']])
                    dataframe_output['ytrue'] = df_test['y'].values
                    if(regressors):
                        dataframe_output['ypred-Prophet(cr)'] = forecast['yhat'].values
                    else:
                        dataframe_output['ypred-Prophet(sr)'] = forecast['yhat'].values

                    #print(dataframe_output)
                    if(regressors):
                        dataframe_output.to_csv('csv_output/'+weather+'_caso'+str(caso)+'_'+str(row[0])+'ProphetCR.csv',index=False)
                    else:
                        dataframe_output.to_csv('csv_output/'+weather+'_caso'+str(caso)+'_'+str(row[0])+'ProphetSR.csv',index=False)

                    '''
                    #forecast = m.predict(future)
                    
                    output_file.write("Estacao:{}\nMAE:{:.3f}\nMAPE:{:.3f}\nMSE:{:.3f}\nRMSE:{:.3f}\n".format(row[1],mean_absolute_error(y_true=df_test['y'], y_pred=forecast['yhat']),mean_absolute_percentage_error(y_true=df_test['y'],y_pred=forecast['yhat']),mean_squared_error(y_true=df_test['y'],y_pred=forecast['yhat'])
                    ,mean_squared_error(y_true=df_test['y'], y_pred=forecast['yhat'], squared=False)))
                    
                
                    #df_p.to_csv(str(row[0])+"_cross_validation.csv")
                    '''
                    try:
                        mkdir("/home/miguel/Documents/TCC/scripts/visualizations")
                    except Exception as e:
                        pass 

                    '''
                    fig1 = m.plot(forecast, uncertainty=True)
                    fig1.savefig("visualizations/"+str(row[0])+'_test.png')
                    fig2 = m.plot_components(forecast)
                    fig2.savefig("visualizations/"+str(row[0])+'_test2.png')
                    f, ax = plt.subplots(1)
                    f.set_figheight(5)
                    f.set_figwidth(15)
                    ax.scatter(df_test['ds'], df_test['y'], color='r')
                    fig3 = m.plot(forecast, ax=ax)
                    fig3.savefig("visualizations/"+str(row[0])+'_test3.png')

                    f, ax = plt.subplots(1)
                    f.set_figheight(5)
                    f.set_figwidth(15)
                    ax.scatter(df_test['ds'], df_test['y'], color='r')
                    fig5 = m.plot(forecast, ax=ax)
                    ax.set_xbound(lower=date(2011,1,1), upper=date(2011,3,30))
                    ax.set_ylim(0, 15)
                    plot = plt.suptitle('2010 Autumn and Winter Forecast vs Actuals')
                    fig5.savefig("visualizations/"+str(row[0])+'_test5.png')
               
        '''
        output_file.close()
        
    else:
        print("Necessary arguments: python3 analisys_data.py <regressores,[0,1]> <testes,[1,5]> <variavel,[Temperatura, Umidade, Precipitacao] ")
        print("#Teste 1 sem Regressors - Ex.: python3 analisys_data.py 0 1 Temperatura")