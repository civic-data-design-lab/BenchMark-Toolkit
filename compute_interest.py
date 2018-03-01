
#import numpy as np
import time
import json
import pickle
import os
import copy
from sets import Set
from optparse import OptionParser
from dateutil.parser import parse
import itertools
from matplotlib import pyplot as plt
import matplotlib.style as style
from math import hypot
#import locations_utility as lu
import compute_locations as cl
from bench_locations import days

"""

What we want to do is to differentiate between the bounding boxes that pop up due to unstable identification of people who sat on benches and the ones
that are identified correctly(people who cut across the field by walking and people who are standing.
Current workflow: Get familiar with the amount of change in rect sizes and stability. Then, derive a computational procedure that can get rid of the unstable.
Method: A gaussian prior on where would the center of a bounding box be given a frame. And if the center of the bounding box is not proximal remove it from the new frame.
Or simply use a well-defined rectangle, and check containment. It might be easy to draw the rectangle on an image editing program(Photoshop)
and transfer the dimensions into the relevant variables in this document
Finally, return the better JSON.

"""

parser = OptionParser()
#parser.add_option("-dir", "--input_dir", dest="input_dir", help="Path to input working directory.", default="/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/tensorflow_models/outputs")
parser.add_option("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/tensorflow_models/temp_pickles")
#parser.add_option("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/tensorflow_models/")
parser.add_option("-d", "--day", dest="day", help="The day to compute values with.", default="0802")
(options, args) = parser.parse_args()
#input_path = os.path.join(options.input_dir, '')
day_text = options.day

# Specify the input filename here

#input_file = 'gopro_files/GoPro_Initial_0802.txt'
input_f = 'gopro_files/GoPro_Initial_%s.txt'%(day_text)

#day = input_file.split('_')[-1].split('.')[0]
day = day_text

with open(input_f) as txt:
	rows = json.load(txt)
	#print rows

# Change the output filename from here

output_filename = "list_of_tuples_%s.p"%(day)
coords_output_filename = "coordinates_%s.p"%(day)
interest_output_filename = "interest_times_%s.p"%(day)
benches_output_filename = "benches_%s.p"%(day)
output_path = os.path.join(options.output_dir, output_filename)
coords_output_path = os.path.join(options.output_dir, coords_output_filename)
interest_output_path = os.path.join(options.output_dir, interest_output_filename)
benches_output_path = os.path.join(options.output_dir, benches_output_filename)
pickle_input_path = output_path


"""
# Read the pickles (if necessary)

with open(pickle_input_path, 'rb') as f:
    peeps = pickle.load(f)

"""

# kwown distance/width of face, reference object in reference image

# Bounding box of reference object in reference image <-- This should be changed
refObj = [216,258,227,265]

KNOWN_DISTANCE_b = 173.5 # Not Face but Body
KNOWN_WIDTH_b = 22 # 23 or 24
KNOWN_DISTANCE_r = 336.0
KNOWN_WIDTH_r = 9.74


refObj_w = 11
reference_image_body_pixel_w = 49

# lat/lon of camera and reference object in space (not used as of 5/23, but will be used to translate from relative space to geographic space) <-- This should be changed
camPos = (766003.20, 2955893.08)
refObjPos = (765996.22, 2955900.69)

#image features
img_width = 800
img_height = 600
img_center = 400
img_cent_orig = (400, 600)


def get_filenames(search_path):
	for (dirpath, _, filenames) in os.walk(search_path):
		for filename in filenames:
			yield filename#os.path.join(dirpath, filename)


def get_rows():
	all_days = []
	filenames = get_file_names(input_path)
	for f in filenames:
		print("reading files and creating dicts...")
		with open(os.path.join(input_path, f)) as txt:
			rows = json.load(txt)
		all_days.append(rows)


def sort_dict(dicto):
	keylist = dicto.keys()
	keylist.sort()
	return keylist

def sort_by_time(dicto_list):
	return sorted(dicto_list, key=lambda k:k['img_timestamp'])

def low_res(dicto_list, res):
	return []

def compute_center(bb):
	center = (bb[2] - bb[0], bb[3] - bb[1])
	return center

def produce_sample_data(start, end, interval):
	return np_linspace(start, end, 50)

#def produce_gaussian_data(mean, variance):
#	return np.random.multivariate_normal(mean, variance, 30)

#def calc_variance(data):
#	nparr = np.asarray(data)
#	return np.var(nparr)

#def make_pdf(point, mean, variance):
#	return multivariate_normal(mean=mean, cov=variance)


# Container boxes, these should be drawn for each day, as there are changes in the position of the camera.
#cb_left_0806 = [119, 262, 219, 322]

# Make a list of container boxes for each day. And check for each cb whether a person is inside.
# e.g. Also put cb's in the leftmost and rightmost strip.

containers = {}

containers['0725']=[[0, 175, 280, 300], [415, 185, 573, 280]]
containers['0726']=[[7, 363, 278, 479], [405, 389, 755, 499]]
containers['0727']=[[116, 243, 304, 315], [385, 248, 535, 326]]
containers['0728']=[[19, 239, 259, 328], [454, 250, 764, 360]]
containers['0731']=[[5, 235, 223, 373], [420, 245, 626, 315]]
containers['0801']=[[0, 240, 209, 380], [108, 223, 304, 341], [317, 234, 428, 310]]
containers['0802']=[[25, 266, 239, 329], [472, 256 , 699, 326]]
containers['0803']=[[0, 249, 239, 389], [493, 223 , 726, 375]]
containers['0804']=[[108, 244, 284, 324], [419, 240 , 513, 290]]
containers['0806']=[[0, 245, 251, 320], [442, 270 , 654, 332]]
containers['charlotte'] = [[462, 305, 585, 430]]
#cb_leftleft_0801 = [0, 264, 209, 393]
#cb_left_0801 = [108, 223, 304, 341]
#cb_right_0801 = [317, 234, 428, 310]



def check_containment_and_ratio_2(direction_cbs, bb):
	center, ratio, w, h = get_center_ratio_w_h(bb)
	return True


def make_num_people_arr(dicto_list):
	return [(d['num_peeps'], d['img_timestamp'], d['bbs']) for d in dicto_list]
	#return [(d['num_peeps'], d['img_timestamp']) for d in dicto_list]
	#return [d['num_peeps'] for d in dicto_list]

def make_dict_list(dicto):
	return dicto.values()

# Recursive function to calculate the max number of people between 0's

def recur_interval(peeps_list, idx, num_list, bb_list):
	max_peep = (0, 0)
	last_idx = 0
	new_list = peeps_list[idx:]
	count = 0
	for i, n in enumerate(new_list):
		if (i == len(new_list) - 1) :
			if n[0] > max_peep[0]:
				max_peep = n
				bb_list.append(n)
				num_list.append(max_peep)
				return (num_list, bb_list)
			elif n[0] == 0:
				num_list.append(max_peep)
				return (num_list, bb_list)
			else :
				bb_list.append(n)
				num_list.append(max_peep)
				return (num_list, bb_list)
		else :
			if n[0] == 0 :
				if new_list[i+1][0] != 0 :
					bb_list.append((0, n[1]))
					last_idx = idx + i + 1
					if max_peep[0] != 0:
						num_list.append(max_peep)
					break

			if n[0] != 0 :
				bb_list.append(n)

			if n[0] > max_peep[0]:
				max_peep = n
				#print max_peep
			count += 1

	return recur_interval(peeps_list, last_idx, num_list, bb_list)


def calc_total_num_people(peeps_list):
	intervals_list = []
	nums_list = []
	bbs_list = []
	n_list = len(peeps_list)/1000
	last = len(peeps_list)%1000
	for i in xrange(n_list+1):
		if (i != n_list):
			first_idx = i*1000
			last_idx = (i+1)*1000
			partial_list = peeps_list[first_idx : last_idx]
		else :
			first_idx = i*1000
			partial_list = peeps_list[first_idx :]

		intervals_list = []
		bounds_list = []
		(num_list, bb_list) = recur_interval(partial_list, 0, intervals_list, bounds_list)
		nums_list.extend(num_list)
		bbs_list.extend(bb_list)

	return (nums_list, bbs_list)


def copy_better_dict(dicto, inval=0, inval_keys=[], timestamp=0, dup_idx_x=0):
	new_dicto = copy.deepcopy(dicto)
	for invalid in inval_keys:
		del new_dicto['bbs'][str(invalid)]

	if timestamp :
		new_dicto['img_timestamp'] = timestamp

	new_dicto['num_peeps'] -= inval 

	if dup_idx_x:
		field = str(dup_idx_x + 1)
		if field not in inval_keys:
			del new_dicto['bbs'][field]
			new_dicto['num_peeps'] -= 1

	return new_dicto


def better_tuples(bbs_arr):
	new_arr = []
	for i, e in enumerate(bbs_arr):
		if len(e) > 2 :
			if e[0] > 1 :
				fields = e[2]
				news =  fields.values()
				bbs = [k['coords'] for k in news]
				new_tuple = (e[0], e[1], bbs)
				new_arr.append(new_tuple)
			else :
				new = e[2].values()
				new_tuple = (e[0], e[1], new[0]['coords'])
				new_arr.append(new_tuple)
		else :
			new_arr.append(e)
	return new_arr



# bbs: (0, u'2017:08:02 19:40:07'), (1, u'2017:08:02 19:40:09', {u'1': {u'coords': [352, 192, 400, 320], u'prob': 89}}), 
# (1, u'2017:08:02 19:40:11', {u'1': {u'coords': [192, 96, 336, 416], u'prob': 95}}), (0, u'2017:08:02 19:49:41'),
# (2, u'2017:08:02 20:13:59', {u'1': {u'coords': [336, 176, 400, 368], u'prob': 92}, u'2': {u'coords': [272, 176, 352, 368], u'prob': 90}})


def parse_relevant_intervals(bbs_arr):

	count_list = []
	new_bbs = []
	count = 0
	part_bbs = []

	for i, e in enumerate(bbs_arr):

		if e[0] == 0:
			if count != 0:
				count_list.append(count)
				bb_tup = [count, part_bbs]
				new_bbs.append(bb_tup)
			count = 0
			part_bbs = []

		else :
			part_bbs.append(e)
			count += 1


	return [e for e in new_bbs if e[0] > 5]
		

def compare_bb(center_a, center_b, height_a, height_b, center_diff, height_diff):
	return (hypot(center_a[0] - center_b[0] , center_a[1] - center_b[1]) < center_diff) #and (abs(height_a - height_b) < height_diff)


# Some new heuristics should be defined here:

def proximity(b, a):
	diff = 20
	center_b = get_center(b)
	center_a = get_center(a)
	return hypot(center_b[0] - center_a[0] , center_b[1] - center_a[1]) < diff

def cache_containment(bb, cache_list):
	for cache in cache_list:
		for c in cache:
			if proximity(bb, c):
				return True


def compute_interestedness(relevant, seconds):

	sticked = 0
	depth = seconds/2 - 1
	times = []
	
	for e in relevant:
		#bb_list = relevant[3][1]
		bb_list = e[1]
		time_list = [ k[1] for k in bb_list ]
		bbs_list = [ k[2] if k[0] > 1 else [k[2]] for k in bb_list ]
		
		# for each bb in bbs_list compare with the bbs in next frame.
		# checking the distance btw centers and diff btw heights.
		# And if they are proximal store the bbs that matched
		# then check the bbs in second next frame to see if there is still a match
		# If there is a match conclude that a person sticked and increment the count

		#for j in cache_list:
		#	print j

		cache_list = []

		for i, bb in enumerate(bbs_list[:-2]):
			count = 0
			if len(bb) == 1:
				if not cache_containment(bb[0], cache_list):
					cache_temp = [] 
					cached = 0
					(sticked, cache, cached) = recur_bb_compare(bb[0], bbs_list[:-2], i+1, sticked, count, depth, cache_temp, cached)
					if cached:
						cache_list.append(cache)
						times.append(time_list[i])


			else:
				for b in bb:
					if not cache_containment(b, cache_list):
						cache_temp = []
						cached = 0
						(sticked, cache, cached) = recur_bb_compare(b, bbs_list[:-2], i+1, sticked, count, depth, cache_temp, cached)	
						if cached:
							cache_list.append(cache)
							times.append(time_list[i])

		#for j in cache_list:
		#	print j
		#	print "-------------------"
		#print time_list
		#print "-------------------"
		#print "-------------------"

	return (sticked, cache_list, times)


def recur_bb_compare(bb, bbs, idx, sticked, count, depth, cache, cached):
	center_dff = 25
	height_dff = 30
	if len(bbs[idx: idx+1]) == 1:
		new = bbs[idx: idx+1][0]
	else :
		new = bbs[idx: idx+1]
	center_a, rat_a, w_a, h_a = get_center_ratio_w_h(bb)
	for j, b in enumerate(new):
		center_b, rat_b, w_b, h_b = get_center_ratio_w_h(b)
		if compare_bb(center_a, center_b, h_a, h_b, center_dff, height_dff) :
			count += 1
			cache.append(b)
			if count > depth:
				sticked += 1
				cached = 1
				break
			else :
				#print sticked
				# Make the recursion here
				(sticked, cache, cached) = recur_bb_compare(b, bbs, idx+j+1, sticked, count, depth, cache, cached)
	
	
	return (sticked, cache, cached)


# [differ_w, differ_h, rat = 2.0, rat_f = 1.5]

tunings = {'0801':[50, 80, 2.4, 1.5], 'charlotte':[40, 60, 2.4, 1.5],'0725': [50, 80, 2.0, 1.5], '0726': [40, 75, 2.2, 1.5], '0727': [32, 60, 2.2, 1.5], '0728':[32, 60, 2.4, 1.5], '0731': [32, 60, 2.2, 1.5],'0802':[32, 60, 2.0, 1.5], '0803':[40, 60, 2.4, 1.5], '0804':[32, 60, 2.0, 1.5], '0806':[32, 60, 2.0, 1.5]}


def get_center_ratio_w_h(bb):
	width = bb[2] - bb[0]
	height = bb[3] - bb[1]
	middleX = width/2
	middleY = height/2
	ratio = float(height)/width
	return ([bb[0] + middleX, bb[1] + middleY], ratio, width, height)

def get_center(bb):
	width = bb[2] - bb[0]
	height = bb[3] - bb[1]
	middleX = width/2
	middleY = height/2
	return (bb[0] + middleX, bb[1] + middleY)

def check_containment_and_ratio(bb, rat, container_list, differ_w = 50, differ_h=80, all_left=False):
	center, ratio, w, h = get_center_ratio_w_h(bb)
	in_any_container = False

	for i, c in enumerate(container_list):
		#from leftmost to rightmost
		in_this = center[0] > c[0] and center[0] < c[2] and center[1] > c[1] and center[1] < c[3]
		in_any_container = in_any_container or in_this
		if  (i == len(container_list) - 1) and in_this and (not all_left) :
			differ_w = 30
			differ_h = 55

	return in_any_container and ((w < differ_w or h < differ_h) or (ratio < rat))
	#return in_any_container and (ratio < rat)


def compute_all_interested(relevant, list_of_seconds, day):
    style.use('fivethirtyeight')
    interested = []
    for i in list_of_seconds:
        final_sticked, final_cache, times = compute_interestedness(relevant, i)
        interested.append(int(final_sticked))

    plt.plot(list_of_seconds, interested, linestyle='-', marker='o', alpha=0.7)
    plt.title('Interestedness July 27')
    plt.xlabel('Time spent in seconds')
    plt.ylabel('Number of people')
    plt.yticks(interested)
    plt.xticks(list_of_seconds)
    plt.show()
    plt.savefig("/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/tensorflow_models/interest_pngs/interested_%s.png"%(day))
    return interested


def loc_proximity(b, a):
	# in feet
	diff = 10 # 5, 6, 7, 10
	return hypot(b[0] - a[0] , b[1] - a[1]) < diff


def compute_clustered_duration_for_benches(times, day):

	# cluster durations for each bench
	# e.g. {'a':(total, [(duration_1, init_time), (duration_2, init_time)])}
	
	#clustered = [0]*6
	clustered = [ [0]*6 for i in range(len(times)) ]
	last_times = [ [0]*6 for i in range(len(times)) ] 
	all_times = {}
	#last_times = [0]*6
	#durations = [0]*6
	first_pair_times = {}
	pairs = {}
	together = {}


	if day == '0728' :
		match = ['4','5','3','6','2','1']
	elif day == '0806':
		match = ['6','5','4','3','1','2']
	else:
		match = ['6','5','4','3','2','1']

	for k, time in enumerate(times):
		#temp_pos = [False]*6
		benches = time[1]
		print time
		
		for i, loc_a in enumerate(benches[:-1]):
			already = False
			for j, loc_b in enumerate(benches[i+1:]):
				pair = match[i] + match[i+j+1]
				if k == 0:
					pairs[pair] = 0
					together[pair] = False
					first_pair_times[pair] = 0
					all_times[pair] = []
				if loc_proximity(loc_b, loc_a):
					print pair
					bench = match[i]
					clustered[k][i] = 1
					clustered[k][i+j+1] = 1
					#last_times[k][i] = time[0]
					#last_times[k][i+j+1] = time[0]
					timestamp = parse(time[0])

					if not (k == len(times)-1) :
						print k

						if not together[pair]:
							#last_times[i] = timestamp
							#last_times[i+j+1] = timestamp
							first_pair_times[pair] = timestamp
							together[pair] = True
							all_times[pair].append(time[0])
						#else :
							#durations[i] += (timestamp - last_pair_times[pair]).total_seconds()
							#durations[i+j+1] += (timestamp - last_pair_times[pair]).total_seconds()
							#last_times[i] = timestamp
	
					else :
						print "I'm here!"
						pairs[pair] += (timestamp - first_pair_times[pair]).total_seconds()
						all_times[pair].append(time[0])
						#durations[i] += (timestamp - last_pair_times[pair]).total_seconds()
						#durations[j+1] += (timestamp - last_times[j+1]).total_seconds()
						#pairs[pair] += (timestamp - last_pair_times[pair]).total_seconds()
				else :
					timestamp = parse(time[0])
					if together[pair]:
						together[pair] = False
						pairs[pair] += (timestamp - first_pair_times[pair]).total_seconds()
						all_times[pair].append(time[0])

	#return zip(match, durations)
	return clustered, all_times, first_pair_times, pairs


def filter_peeps(rows, day):
	
	better = []
	visitor_sum = 0

	# For now, the differ constant is determined by eyeballing the widths of bounding boxes.
	# It can alternatively be determined by using a gaussian. 
	# Rat constant is a useful heuristic to differentiate between the bb's of people who are sitting down and who are standing up.

	differ_w = tunings[day][0]
	differ_h = tunings[day][1]
	rat = tunings[day][2] # 2.0, 
	rat_f = tunings[day][3] # 1.5

	sorted_keys = sort_dict(rows)

	for idx, key in enumerate(sorted_keys):
		row = rows[key]
		timecandi = row['img_timestamp']
		timestamp = parse(timecandi)
		hour, minute, second = timestamp.hour, timestamp.minute, timestamp.second
		peeps = row['num_peeps']
		visitor_sum += peeps
		bbs = row['bbs']

		invalid = 0
		invalid_keys = []
		dup_index_x = 0
		all_Xs = []
		all_Ys = []
		new_bbs = {}

		# With this we get rid of people who are far away.
		
		for i, (k, v) in enumerate(bbs.items()):
			invalid_already = False
			if v :
				coords = v['coords']
				all_Xs.append(coords[0])
				center, ratio, w, h = get_center_ratio_w_h(coords)

				#if check_containment_and_ratio(coords, rat, cb_left_0801, cb_right_0801):
				if check_containment_and_ratio(coords, rat, containers[day], all_left=True):
					invalid += 1
					invalid_keys.append(k)
					invalid_already = True

				if (w < differ_w or h < differ_h) and not invalid_already :
					#print "this is far away!"
					invalid += 1
					invalid_keys.append(k)
					invalid_already = True

				if ratio < rat_f and not invalid_already :
					invalid += 1
					invalid_keys.append(k)
					

		# Here check get rid of the double counts of one individual
		dups_x = [i for i, x in enumerate(all_Xs) if all_Xs.count(x) > 1]

		if len(dups_x) > 1 :
			dup_index_x = dups_x[1]

		better.append(copy_better_dict(row, invalid, invalid_keys, 0, dup_index_x))


	return better

def remove_bad_stops(bbs_arr):

	#concatenate intervals when a stop is not surrounded by values of 1
    #accounts for people who may be double counted as a result of an issue with the machine recognizing people

	new_bbs = []
	templist = []
	append_to_next = False
	prev = bbs_arr[0][1]
	if prev[-1] != 1:
		prev = []
	do_next = True

	for f in [e[1] for e in bbs_arr]:
		for ef in f:
			templist.append(ef[0])
		if do_next and templist[0] == 1:
			bb_tup = [len(prev), prev]
			new_bbs.append(bb_tup)
			prev = f
		do_next = False
		if append_to_next or templist[0] != 1:
			prev = prev + f
			if templist[-1] == 1:
				do_next = True
		elif templist[-1] == 1:
			do_next = True
		if templist[-1] != 1:
			append_to_next = True
		else:
			append_to_next = False
		templist = []
	if append_to_next or do_next:
		bb_tup = [len(prev), prev]
		new_bbs.append(bb_tup)
	if new_bbs[0][1] == []:
		del new_bbs[0]

	return new_bbs

# We would like to categorize images by day
def main():

	# Write the loop when you receive multiple output txt files

	better = filter_peeps(rows, day)

	#print better
	peeps_arr = make_num_people_arr(sort_by_time(better))
	#del peeps_arr[3911:4928]

	# We want to parse this pattern intelligently and to ignore 0's that squeezed between crowd as a parse signal
	#print [a[0] for a in peeps_arr] 

	peeps, bbs = calc_total_num_people(peeps_arr)
	
	#for pair in peeps :
	#	print(pair[0]) 
	#print [pair[0], pair[1] for pair in peeps]
	#print sum([pair[0] for pair in peeps])
	#with open(output_path, 'wb') as f:
	#	pickle.dump(peeps, f)

	#print bbs
	#print peeps_arr
	#print bbs
	
	#print [(triple[0],triple[1]) for triple in bbs]
	tups = better_tuples(bbs)
	#print [(e[0]) for e in tups]

	relevant = parse_relevant_intervals(tups)

	#relevant_with_bad_stops = parse_relevant_intervals(tups)
	#relevant = remove_bad_stops(relevant_with_bad_stops)

	#for f in [e[1] for e in relevant]:
	#	print "stop"
	#	print [ef[0] for ef in f]
	
	#print "----------------------"
	#print "----------------------"
	#print [e[0] for e in relevant]

	#final_sticked, final_cache, times = compute_interestedness(relevant, 6)

	#print final_sticked, final_cache, times
	#print final_sticked
	#print times
	#print bbs

	#for pair in bbs: 
	#	print bbs
	
	locs = cl.plot_and_publish_locs(peeps_arr, benches_output_path, day, True)
	clustered, all_times, first_pair_times, pairs = compute_clustered_duration_for_benches(locs, day)
	keys = pairs.keys()
	keys.sort()
	for k in keys:
		print k, pairs[k], first_pair_times[k], all_times[k]

	#for i in range(6):
	#	print last_times[i*len(locs):(i+1)*len(locs)]

	#for i in range(6):
	#	print last_times[i*len(locs):(i+1)*len(locs)]
		

	#a = [pair for pair in pair_keys if "a" in pair]

	#hop = compute_all_interested(relevant, [6, 8, 10, 12, 20, 30, 40], day)

	#with open(interest_output_path, 'wb') as f:
	#		pickle.dump(times, f)
	


if __name__ == '__main__':
	main()






	




