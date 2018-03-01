# python file containing functions for ped detection, image processing, etc.
# meant to serve as toolkit for all methods and scripts written thus far

import numpy as np
import imutils
import cv2
from imutils.object_detection import non_max_suppression
import math
import glob
import os
import matplotlib
import csv
import itertools
matplotlib.use("TkAgg")  # this is specifically for plotting with matplotlib on macosx el capitan


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

# determines which images have not yet been processed and adds them to the queue
def remove_dups(imgDir, imgList):

    with open(imgList, 'rb') as f:
        reader = csv.reader(f)
        img_list = list(reader)

    # cleans up list of already processed images
    il = list(itertools.chain.from_iterable(img_list))
    imgDir.sort()
    il.sort()

    # determines which images have not yet been processed and adds to list
    culled_images = [x for x in imgDir if x not in il]

    return culled_images

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

def find_faces(image, classifier, refObj, refObjMid, img_cent_orig):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converts image to grayscale

    # TODO: experiment with changing image contrast, etc. to optimize classification
    face = classifier.detectMultiScale(gray, 1.3, 5) # classifies the image based on the classifier file you pass

    print "number of faces: ", len(face) # returns number of faces identified in image

    # if no features are found, an empty tuple instead of an array is passed from the classifier
    # this passes dummy points through so as to not have other functions fail later
    if type(face) is tuple:
        # TODO: find a better way to deal with negative images
        rects = [[0, 0, 1, 1]]
        midpoint = [[0.5, 0.5]]
        return rects, midpoint
    else:

        rects = [] # all of the rectangles for all of the faces in the image
        midpoint = [] # all of the midpoints for all of the rectangles in the image

        for (x, y, w, h) in face:
            i = 0

            # defines bounding box coordinates
            rect = [x, y, x + w, y + h]

            # draws bounding box on image
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # finds midpoint of bounding box
            rectsMid = ((((rect[2] - rect[0]) / 2) + rect[0]), (((rect[3] - rect[1]) / 2) + rect[1]))

            # draws bounding box of reference object (not strictly necessary)
            rectRef = cv2.rectangle(image, (refObj[0], refObj[1]), (refObj[2], refObj[3]), (0, 0, 255), 2)

            # writes midpoints and bounding boxes to arrays (for cases when there are multiple detections per image)
            midpoint.append(rectsMid)
            rects.append(rect)

            # draws a line between reference object and face
            distLine = cv2.line(image, refObjMid, rectsMid, (0, 0, 255), 5)

            # draws a line between face and camera
            camFaceLine = cv2.line(image, img_cent_orig, rectsMid, (0, 0, 255), 5)

            # draws a line between camera and reference object
            camRefLine = cv2.line(image, img_cent_orig, refObjMid, (0, 0, 255), 5)
            i += 1

    return rects, midpoint

# calibrates camera based on know size of objects, and known distances.
# all of the distance calculations depend on a verified distance reading, recorded in a reference image
def get_focal_length(faceWidth, refWidth, KNOWN_DISTANCE_r, KNOWN_DISTANCE_f, KNOWN_WIDTH_f, KNOWN_WIDTH_r):
    focalLength_f = (faceWidth * KNOWN_DISTANCE_f) / KNOWN_WIDTH_f
    focalLength_r = (refWidth * KNOWN_DISTANCE_r) / KNOWN_WIDTH_r

    return focalLength_f, focalLength_r

# measures observed width of face, based on size of bounding box
def get_face_width(marker):
    faceWidth = (marker[2] - marker[0])
    return faceWidth

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