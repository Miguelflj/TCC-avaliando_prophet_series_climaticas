import os
path = '/home/miguel/Documents/TCC/Estados'
init_states = ['AC','AL','AM','AP','BA','CE', 'DF', 'ES', 'GO', 'MA', 'MG', 'MS', 'MT', 'PA','PB','PE', 'PI', 'PR', 'RJ','RN','RO','RR','RS','SC','SE','SP','TO']


for stt in init_states:
    file = open(path+'/'+'headers_info_'+stt+'.txt','r')
    content = file.readlines()
    for i in range(0,len(content),10):
        name = content[i]
        cod = content[i+1]
        lat = content[i+2]
        long = content[i+3]
        alt = content[i+4]
        sit = content[i+5]
        dat_init = content[i+6]
        dat_final = content[i+7]
        new_name = "_".join(name.split()[1:])
        os.makedirs(path+'/'+stt+'/'+'header', exist_ok=True)
        new_file = open(path+'/'+stt+'/'+'header/'+new_name+'_'+stt+'.txt','w')    
        new_file.write("{}{}{}{}{}{}{}{}".format(name,cod,lat,long,alt,sit,dat_init,dat_final))
        new_file.close()
    file.close()
