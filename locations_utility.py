
## Functions adapted from Mario Giampieri's work.

import numpy as np
import time
import json
import pickle
import os
import copy
from optparse import OptionParser
from dateutil.parser import parse
import itertools
import matplotlib
import math


# list of images that have been processed
#img_list = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/processed_images.csv"
#img_list= "/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/processed_images.csv"
# directory of images to classify
#img_dir = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/images/"
img_dir = "/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/images/"


# image with object we're interested in at a known distance
#reference_img = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/reference_image/G0082212.JPG"
reference_img="/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/reference_image/ege_face.JPG"


def tuples_for_locations(tuple_list):
    listo = []
    for d in tuple_list:
        if d[0] > 0 :
            vals = d[2]
            if len(vals) > 1 :
                bbs = []
                fields = vals.values()
                for val in fields:
                    bbs.append(val['coords'])
            else :
                field = vals.values()
                bbs = field[0]['coords']

            listo.append((d[0], d[1], bbs))
        else:
            listo.append((d[0], d[1]))
                    
    return listo


# gets jpegs out of a directory
def get_jpgs(imgDir):
    images = []
    os.chdir(imgDir)
    for file in os.listdir(imgDir):
        if file.lower().endswith(".jpg"):
            images.append(file)

    return images

# gets pngs out of a directory
def get_pngs(imgDir):
    images = []
    os.chdir(imgDir)
    for file in os.listdir(imgDir):
        if file.lower().endswith(".png"):
            images.append(file)
    return images

# gets midpoint and width of reference object
def get_refObj_statistics(refObj):
    refObjMid = ((((refObj[2] - refObj[0]) / 2) + refObj[0]), (((refObj[3] - refObj[1]) / 2) + refObj[1]))
    refWidth = refObj[2] - refObj[0]

    return refObjMid, refWidth

# returns properties of an image for use in calculating angles in get_coord_pair below
def get_image_statistics(image):
    height, width, channels = np.shape(image)
    img_center = width / 2
    img_cent_orig = (img_center, height)
    return height, width, img_center, img_cent_orig

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth


# calibrates camera based on know size of objects, and known distances.
# all of the distance calculations depend on a verified distance reading, recorded in a reference image
def get_focal_length(faceWidth, refWidth, KNOWN_DISTANCE_r, KNOWN_DISTANCE_f, KNOWN_WIDTH_f, KNOWN_WIDTH_r):
    focalLength_f = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r

    return focalLength_f, focalLength_r

# measures observed width of face, based on size of bounding box
def get_body_width(bb):
    bodyWidth = bb[2] - bb[0]
    return bodyWidth

# measures observed width of face, based on size of bounding box
def bb_center(bb):
	width = bb[2] - bb[0]
	height = bb[3] - bb[1]
	middleX = width/2
	middleY = height/2
	ratio = float(height)/width
	return (bb[0] + middleX, bb[1] + middleY)

# finds distance between face and camera, ref object and camera, and face and reference object
def get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r):

    # distance from face to camera in inches (it's inches because that's the unit that the reference dimensions are given)
    inches_f = distance_to_camera(KNOWN_WIDTH_f, focalLength_f, faceWidth)
    inches_r = distance_to_camera(KNOWN_WIDTH_r, focalLength_r, refWidth)

    # distances converted to feet
    feet_f = inches_f / 12
    feet_r = inches_r / 12

    distBetween = math.sqrt(feet_f ** 2 + feet_r ** 2 - (2 * feet_f * feet_r * math.cos(angle)))

    return inches_f, inches_r, feet_f, feet_r, distBetween

# uses distance from objects and image statistics to calculate x,y coordinates relative to camera position
def get_coord_pair(dist_f, midpoint, dist_r, img_center, height, refObjMid):

    # writes camera position coordinates
    camCoord = (img_center, height)

    # finds sides of ref obj triangle in pixels
    refObjMidx = (img_center - refObjMid[0]) # distance in x-plane from bbox centroid to image center
    refObjMidy = (height - refObjMid[1]) # distance in y-plane from bbox centroid to image bottom

    # finds length of hypotenuse of triangle in pixels to be able to scale
    # relative to observed distance
    distPix_r = math.sqrt(refObjMidx**2+refObjMidy**2)

    # finds scale factor to multiply pixel sides by
    scale_r = dist_r / distPix_r

    # scales x and y sides
    refObjMidx_feet = scale_r * refObjMidx
    refObjMidy_feet = scale_r * refObjMidy

    # writes sides of triangle to a data object
    refObj_a_b_c = refObjMidx_feet, refObjMidy_feet, dist_r

    # repeats, the above for face obj triangle in pixels
    faceObjMidx = -(img_center - midpoint[0])

    faceObjMidy = (height - midpoint[1])
    distPix_f = math.sqrt(faceObjMidx ** 2 + faceObjMidy ** 2)

    # finds scale factor to multiply pixel sides by
    scale_f = dist_f / distPix_f

    faceObjMidx_feet = scale_f * faceObjMidx
    faceObjMidy_feet = scale_f * faceObjMidy

    faceObj_a_b_c = faceObjMidx_feet, faceObjMidy_feet, dist_f

    refObjCoord = ((img_center- refObjMid[0]), dist_r)

    # midpoint of face bounding box
    objCoord = ((img_center- midpoint[0]), dist_f)

    return camCoord, refObjCoord, objCoord, refObj_a_b_c, faceObj_a_b_c




