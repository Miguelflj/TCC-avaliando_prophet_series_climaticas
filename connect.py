
import psycopg2


class DATABASE:
    def __init__(self, app, user, password):
        
        try:
            self.conn = psycopg2.connect(database=app, user=user,password=password, host="localhost", port="5432")
            print("Conectado ao banco com sucesso.")

            self.on = True
        except:
            print("Falha na conexão ao banco.")
            self.on = False
    
    def create_db(self, sql):
        curr = self.conn.cursor()
        try:
            curr.execute(sql)
            print("OK.")
        except Exception as e:
            print('error', e)
        self.conn.commit()
    def disconnect(self):
        self.conn.close()

    
    def insert_estacoes(self, cod, nome, sg, long, lat, alt):
        curr = self.conn.cursor()
        sql = """
        INSERT INTO tbl_estacoes(cod_estacao,nome_estacao,sg_estado,long,lat,alt)
        VALUES({},\'{}\',\'{}\',{},{},{});
        """.format(cod,nome,sg,long,lat,alt)
        try:
            curr.execute(sql)
            #print("ok")
        except Exception as e:
            print('error', e)
        self.conn.commit()
    
    def insert_dados_diarios(self,cod, data, prep,temp,umid, imput):
        curr = self.conn.cursor()
        sql = """
        INSERT INTO tbl_dadosdiarios (cod, data, precipitacao, temperatura, umidade, imputado)
        VALUES({},\'{}\',{},{},{},{});
        """.format(cod,data,prep,temp,umid,imput)
        try:
            curr.execute(sql)
            #print("ok")
        except Exception as e:
            print('error', e)
        self.conn.commit()

    def get_estations(self):
        curr = self.conn.cursor()
        query_sql = """ 
            select * from tbl_estacoes;
        """

        try:
            curr.execute(query_sql)
            print("Selecting rows from mobile table using cursor.fetchall")
            data_query = curr.fetchall()
            return  data_query
        except Exception as e:
            print('error', e)

    def get_data_of_estations(self, cod):
        curr = self.conn.cursor()
        query_sql = """ 
            select cod, data, precipitacao, temperatura, umidade, imputado from tbl_dadosdiarios where cod = \'{}\';
        """.format(cod)
        try:
            curr.execute(query_sql)
            #print("Selecting rows from mobile table using cursor.fetchall")
            data_query = curr.fetchall()
            return  data_query
        except Exception as e:
            print('error', e)
        
   
    def get_nulls(self, cod):
        curr = self.conn.cursor()
        query_sql = """ 
            select cod, data, precipitacao, temperatura, umidade, imputado from tbl_dadosdiarios where cod = \'{}\' and ( (precipitacao is NULL) or (temperatura is NULL) or (umidade is NULL) );
        """.format(cod)
        try:
            curr.execute(query_sql)
            #print("Selecting rows from mobile table using cursor.fetchall")
            data_query = curr.fetchall()
            return  data_query
        except Exception as e:
            print('error', e)

    def set_imput_data(self, data,cod, prep, temp,umid, imput):
        curr = self.conn.cursor()
        query_sql = """
            UPDATE tbl_dadosdiarios SET (Precipitacao,Temperatura,Umidade,Imputado) = ({},{},{},{}) WHERE cod = {} AND data = \'{}\';
        """.format(prep,temp,umid,imput,cod,data)
        print(query_sql)
        try:
            curr.execute(query_sql)
        except Exception as e:
            print('error', e)   
        print(self.conn.commit())
 
table_estacoes = """
    CREATE TABLE IF NOT EXISTS tbl_estacoes (
            Cod_estacao INT CONSTRAINT pk_cod_estacao PRIMARY KEY,
            Nome_estacao varchar(35) UNIQUE NOT NULL,
            SG_estado varchar(5),
            Long float,
            Lat float,
            Alt float
    )
"""
table_dados_diarios_estacao = """
    CREATE TABLE IF NOT EXISTS tbl_dadosDiarios (
        control serial primary key,
        Cod INT NOT NULL ,
        Data date NOT NULL,
        Precipitacao float,
        Temperatura float,
        Umidade float,
        Imputado INT,
        FOREIGN KEY (Cod) REFERENCES tbl_estacoes (Cod_estacao)
    )
"""


tables = ['estacoes','dadosDiarios']
if __name__== '__main__':
    base = DATABASE("estacoes","postgres","123")
    if(base.on):
   
        base.create_db(table_estacoes)
        base.create_db(table_dados_diarios_estacao)
       
        #base.create_db('ALTER TABLE IF EXISTS tbl_dadosDiarios RENAME COLUMN Prepicitacao TO Precipitacao')        
        base.disconnect()
