import os
import converter

# configuration 
dataPath = "./data/"

fileNames = os.listdir(dataPath)
for file in fileNames:
    # print(file[-4:])
    if(file[-4:]=='.mid'):
        prefix = file[:-4]
        suffix = '.txt'
        converter.miditodata(dataPath + prefix + '.mid', dataPath + prefix + '.txt')
