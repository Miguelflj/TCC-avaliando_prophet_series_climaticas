from email import header
from os import listdir, mkdir,rename
from os.path import isfile, join
import pandas as pd

#dir = '../Estados/'
#dir = '/home/miguel/Documents/Test/'
dir = '/home/miguel/Documents/TCC/Estados/'
sgs = ['AC','AL','AM','AP','BA','CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA','PB','PE', 'PI', 'PR', 'RJ','RN','RO','RR','RS','SC','SE','SP','TO']
 
for sg in sgs:
    header_arq = open(dir+'/'+"headers_info_"+sg+".txt",'w+')
    files = [f for f in listdir(dir+sg) if isfile(join(dir+sg,f))]
    for f in files:
        path_file = dir+sg+'/'
        arq = open(path_file+f,'r+')
        content = arq.readlines()
        arq.close()
        new_file = open(path_file+f,'w+')
        header = content[:10]
        header_arq.writelines(header)
        data = content[10:]
        new_file.writelines(data)
        new_file.close()
    header_arq.close()