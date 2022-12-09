
from os import RWF_NOWAIT, listdir, rename
from os.path import isfile, join
from geopy.geocoders import Nominatim

states = {
   "Acre" : "AC",
   "Alagoas":"AL",
   "Amapá":"AP",
   "Amazonas":"AM",
   "Bahia":"BA",
   "Ceará":"CE",
   "Distrito Federal":"DF",
   "Espírito Santo":"ES",
   "Goiás":"GO",
   "Maranhão":"MA",
   "Mato Grosso":"MT",
   "Mato Grosso do Sul":"MS",
   "Minas Gerais":"MG",
   "Pará":"PA",
   "Paraíba":"PB",
   "Paraná":"PR",
   "Pernambuco":"PE",
   "Piauí":"PI",
   "Rio de Janeiro":"RJ",
   "Rio Grande do Norte":"RN",
   "Rio Grande do Sul":"RS",
   "Rondônia":"RO",
   "Roraima":"RR",
   "Santa Catarina":"SC",
   "São Paulo":"SP",
   "Sergipe":"SE",
   "Tocantins":"TO"
}



path = "/home/miguel/Documents/TCC/Estacoes"
files = [f for f in listdir(path) if isfile(join(path, f))]
for f in files:
    content = open(path+ "/" +f).readlines()
    name = content[0].split()[1:]
    lat = content[2].split()[1]
    long = content[3].split()[1]
    geolocator = Nominatim(user_agent="GetLoc")
    print("Name:{} Lat:{} Long:{}".format(name, lat, long))
    location = geolocator.reverse(lat+","+long)
    address = location.address.split(",")
    if(address[-2].strip().replace("-","").replace(".","").isnumeric()):
        state_name = address[-4].strip()
    else:
        state_name = address[-3].strip()
    state_sg = states[state_name]
    string_name = "_".join(name)
    new_name = "{}_{}.csv".format(string_name,state_sg)
    rename(path+"/"+f,path+"/"+new_name)