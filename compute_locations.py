import numpy as np
from skimage.transform import ProjectiveTransform
import time
import json
import pickle
import os
from matplotlib import pyplot as plt
from math import hypot
import locations_utility as lu
from bench_locations import tuples_for_benches, days


refs = {'0802':[216,258,227,265], '0725':[77, 216, 97, 228], '0726':[0,366,27, 354], '0727':[173,250,184,257], '0728':[229,265,240,272], '0731':[218,249,229,256], '0801':[299,263,310,270], '0803':[252,260,263,267], '0804':[189,244,200,251], '0806':[206,258,217,265]}
refObj = [216,258,227,265] # Bounding box of reference object in reference image <-- This should be changed

KNOWN_DISTANCE_b = 173.5 # 173.5 # Not Face but Body <-- Change this parameter if locations look abnormal
KNOWN_WIDTH_b = 30 # 22 / 23 / 24 or try even 30 <-- Change this parameter if locations look abnormal
KNOWN_DISTANCE_r = 336.0 # <-- This should be accurate
KNOWN_WIDTH_r = 9.74 # <-- Accurate as well
KNOWN_DISTANCE_be = 172.8 
KNOWN_WIDTH_be = 29 # inches


ref_ws = {}

refObj_w = 11 # Change this if day changes
reference_image_body_pixel_w = 49 # Change this if day changes
bench_pixel_w = 69
# lat/lon of camera and reference object in space (not used as of 5/23, but will be used to translate from relative space to geographic space) <-- This should be changed
#camPos = (766003.20, 2955893.08)
refObjPos = (42.361492, -71.089583)
camPos = (42.361486, -71.089480)

#image features
img_width = 800
img_height = 600
img_center = 400
img_cent_orig = (400, 600)

def get_ref_width(ref):
	return ref[2]-ref[0]

def plot_and_publish_locs(peeps_arr, coords_output_path, day, benches=False):

	refObj = refs[day]
	#print(refObj)

	focalLength_be, focalLength_r = lu.get_focal_length(bench_pixel_w, get_ref_width(refObj), KNOWN_DISTANCE_r, KNOWN_DISTANCE_be, KNOWN_WIDTH_be, KNOWN_WIDTH_r)
	refObjMid = lu.bb_center(refObj)
	angle = refObjMid[0]*90/img_center

	faceX = []
	faceY = []
	locs = []

	#bottom_left = [-5, 0]
	#top_left = [-15, 30]
	#top_right = [15, 30]
	#bottom_right  = [5, 0]

	bottom_left = [-15, 0]
	top_left = [-15, 50]
	top_right = [15, 50]
	bottom_right  = [15, 0]

	t = ProjectiveTransform()

	base = np.asarray([bottom_left, top_left, top_right, bottom_right])
	transform_to = np.asarray([[-25, 0], [-25, 50], [25, 50], [25, 0]])
	if not t.estimate(base, transform_to): raise Exception("estimate failed")
	
	#base = np.asarray([bottom_left, top_left, top_right, bottom_right])
	#transform_to = np.asarray([[-15, 0], [-15, 30], [15, 30], [15, 0]])
	#if not t.estimate(base, transform_to): raise Exception("estimate failed")

	if not benches :
		loc_tuples = lu.tuples_for_locations(peeps_arr)
	else :
		tuples = tuples_for_benches(days)
		#for i in tuples:
		#	print i
		loc_tuples = [i[1] for i in tuples if i[0] == day][0]
		#print loc_tuples

	for j, item in enumerate(loc_tuples):

		if not benches :

			if item[0] != 0:
				bbs = item[2]
				#print bbs

				if type(bbs[0]) == int :
					bodyWidth = lu.get_body_width(bbs)
					inches_be, inches_r, feet_be, feet_r, distBetween = lu.get_distance(bodyWidth, get_ref_width(refObj), angle, KNOWN_WIDTH_be, KNOWN_WIDTH_r, focalLength_be, focalLength_r)
					cam, ref, face, refTriangle, faceTriangle = lu.get_coord_pair(feet_be, lu.bb_center(bbs), feet_r, img_center, img_height, refObjMid)
					transformed = t([faceTriangle[0], faceTriangle[1]])[0]
					latlon = (transformed[0], transformed[1])
					faceX.append(int(transformed[0]))
					faceY.append(int(transformed[1]))
				
				else :
					latlon = []
					facex = []
					facey = []
					for bb in bbs:
						bodyWidth = lu.get_body_width(bb)
						inches_be, inches_r, feet_be, feet_r, distBetween = lu.get_distance(bodyWidth, get_ref_width(refObj), angle, KNOWN_WIDTH_be, KNOWN_WIDTH_r, focalLength_be, focalLength_r)
						cam, ref, face, refTriangle, faceTriangle = lu.get_coord_pair(feet_be, lu.bb_center(bb), feet_r, img_center, img_height, refObjMid)
						transformed = t([faceTriangle[0], faceTriangle[1]])[0]
						latlon.append((transformed[0], transformed[1]))
						facex.append(int(transformed[0]))
						facey.append(int(transformed[1]))
					faceX.append(facex)
					faceY.append(facey)
					

			else:
				latlon = (0, 0)
				faceX.append(' ')
				faceY.append(' ')

			locs.append((str(item[1]), latlon))

		else :
			latlon = []
			facex = []
			facey = []
			labels = item[1].keys()
			labels.sort()
			for i in labels:
				bb = item[1][i]
				bodyWidth = lu.get_body_width(bb)
				inches_be, inches_r, feet_be, feet_r, distBetween = lu.get_distance(bodyWidth, get_ref_width(refObj), angle, KNOWN_WIDTH_be, KNOWN_WIDTH_r, focalLength_be, focalLength_r)
				cam, ref, face, refTriangle, faceTriangle = lu.get_coord_pair(feet_be, lu.bb_center(bb), feet_r, img_center, img_height, refObjMid)
				transformed = t([faceTriangle[0], faceTriangle[1]])[0]
				latlon.append((transformed[0], transformed[1]))
				facex.append(int(transformed[0]))
				facey.append(int(transformed[1]))
			faceX.append(facex)
			faceY.append(facey)

			locs.append((str(item[0]), latlon, labels))
		

		#results[str(tup[1])].append(latlon)

		#plt.set_cmap("Reds")
		# scales color ramp based on number of images. darker colors are newer observations
		#numImg = len(images)
		#c = [float(i) / float(numImg), 0.0, float(numImg - i) / float(numImg)]
		
		plt.close()
		#print str(item[0])
		plt.title("%s"%(str(item[0])))
		plt.axis([-25, 25, 0, 50])
		if faceX[j] != ' ' :
			plt.scatter(faceX[j], faceY[j], s=100, edgecolors='black')

		#if benches :
		#	for label, x, y in zip(labels, faceX, faceY):
		#		plt.annotate(
		#			label,
		#			xy=(x, y), xytext=(-20, 20),
		#			textcoords='offset points', ha='right', va='bottom',
		#			bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
		#			arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

		plt.grid()
		plt.gca().set_aspect('equal', adjustable='box')
		#print item[0]
		#print j
		#plt.show()
		#plt.savefig("bench_pngs/%s.png"%(item[0]))
		

	#for i in locs:
	#	print i		
	with open(coords_output_path, 'wb') as f:
			pickle.dump(locs, f)

	return locs		

