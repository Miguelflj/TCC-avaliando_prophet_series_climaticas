from os import listdir, mkdir,rename
from os.path import isfile, join


sgs = ['CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA','PB','PE', 'PI', 'PR', 'RJ','RN','RO','RR','RS','SC','SE','SP','TO']
base_path = "/home/miguel/Documents/TCC/Estados/"


for sg in sgs:
    files = [f for f in listdir(base_path+sg) if isfile(join(base_path+sg,f))]
    for f in files:
        new_name = f[2:]
        rename(base_path+sg+"/"+f,base_path+sg+"/"+new_name)