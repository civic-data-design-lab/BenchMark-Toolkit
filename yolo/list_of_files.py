from os import listdir
from os.path import isfile, join

mypath = "images/"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in onlyfiles:
	with open("train.txt", "a") as myfile:
		myfile.write("/home/ubuntu/darknet/data/gehl/"+f+"\n")


