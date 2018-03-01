import os, random, sys
import shutil

num_images = 655

root_dir = '/Users/egeozin/desktop/civic_data/gehl/vision/yolo/'
directory = '/Users/egeozin/desktop/civic_data/gehl/vision/yolo/sample/'
dir_list = os.listdir(directory)

images_list = []
txt_list = []
selected = []

for i in dir_list:
	if 'jpg' in i:
		images_list.append(i)
	if 'txt' in i:
		txt_list.append(i)


train = (num_images/10)*7
test = num_images - train

random.shuffle(images_list)
for i in range(test):
	img = images_list[i]
	first = img[0:img.find('.')]
	#first = img.compile(r"^(.*?)\..*")
	print first
	selected.append(first) 

#print train
#print test
#print images_list

#if not os.path.exists(root_dir+'test_set/'):
#    os.makedirs(root_dir+'test_set/')
#
#if not os.path.exists(root_dir+'train_set/'):
#    os.makedirs(root_dir+'train_set/')

for root, dirs, files in os.walk(directory):
   for f in files:
   		if 'jpg' in f:
   			if os.path.splitext(f)[0] in selected:
   				#shutil.copy2(os.path.join(directory, f), 'test_set/')
   				with open("test.txt", "a") as myfile:
					myfile.write("/home/ubuntu/darknet/data/gehl/"+f+"\n") 
			else:
				#shutil.copy2(os.path.join(directory, f), 'train_set/')
				with open("train.txt", "a") as myfile:
					myfile.write("/home/ubuntu/darknet/data/gehl/"+f+"\n") 