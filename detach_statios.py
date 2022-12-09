
from os import listdir, mkdir
from os.path import isfile, join
import shutil

source = "/home/miguel/Documents/TCC/Estados/"
destination = "/home/miguel/Documents/TCC/Estados/"


files = [f for f in listdir(source) if isfile(join(source, f))]
for f in files:
    state_sg = f.replace(".csv", "").split("_")[-1]
    shutil.move(source+f,destination+state_sg+"/"+f)
   