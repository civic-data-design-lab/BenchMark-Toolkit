# a file to refactor distance_between.py and gehl_detect.py; goal is to move most functions to gehl_detect.py and have this file primarily hold parameter definitions

# import libraries
import imutils
import cv2
#import matplotlib
#matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt
import gehl_detect as gd
import csv
import json
from datetime import datetime

# TODO: combine profile and frontal face classification into one classifier (using human head size for profile)

# list of images that have been processed
#img_list = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/processed_images.csv"
img_list= "/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/processed_images.csv"

# directory of images to classify
#img_dir = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/images/"
img_dir = "/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/images/"


# image with object we're interested in at a known distance
#reference_img = "/Users/Mario/Documents/mit-github-projects/gehl-cddll/global/reference_image/G0082212.JPG"
reference_img="/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/reference_image/ege_face.JPG"

# the classifier we're using
face_cascade= cv2.CascadeClassifier("/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/opencv/classifiers/haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier("/Users/egeozin/Desktop/civic_data/gehl/gehl_vision/mario/opencv/classifiers/haarcascade_profileface.xml")

# pulls jpegs out of directory and writes to list
images = gd.get_jpgs(img_dir)

# goes through csv file of all images processed thus far and removes already processed images from the queue
images = gd.remove_dups(images, img_list)

# bounding box of reference object in reference image
refObj = [408,1019,506,1100]

# returns reference object statistics from image
refObjMid, refWidth = gd.get_refObj_statistics(refObj)

# kwown distance/width of face, reference object in reference image // inches
KNOWN_DISTANCE_f = 173.5
KNOWN_WIDTH_f = 22 # 23 or 24
KNOWN_DISTANCE_r = 324.0
KNOWN_WIDTH_r = 9.74

# lat/lon of camera and reference object in space (not used as of 5/23, but will be used to translate from relative space to geographic space)
camPos = (766003.20, 2955893.08)
refObjPos = (765996.22, 2955900.69)

# reads reference image
image = cv2.imread(reference_img)

# define reference image statistics
height, width, img_center, img_cent_orig = gd.get_image_statistics(image)

# get angle between midpoint of reference object and center of image
angle = refObjMid[0]*90/img_center

# finds marker in reference image
marker, midpoint = gd.find_faces(image, face_cascade, refObj, refObjMid, img_cent_orig)

# returns width of face in reference image. note that face is at known distance- this is key
faceWidth = gd.get_face_width(marker[0])

# calibrates camera based on known parameters and observed parameters from reference image
focalLength_f, focalLength_r = gd.get_focal_length(faceWidth, refWidth, KNOWN_DISTANCE_r, KNOWN_DISTANCE_f, KNOWN_WIDTH_f, KNOWN_WIDTH_r)

# calculates distance between face, refobj, and camera in reference image
inches_f, inches_r, feet_f, feet_r, distBetween = gd.get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r)

# sets up lists of face coordinates for positioning in loop
faceX = []
faceY = []

# creates a dictionary to hold image bounding boxes
results = {}

# sets up color scale generator counter in loop
j = []

i = 1
# with condition to write results to json file with timestamp
with open("data_%s.json" % str(datetime.now()), "w") as resOut:
    # with condition sets up adding processed files to csv
    with open(img_list, 'a') as file:
        textwriter = csv.writer(file, delimiter=",")
        for each in images:
            print "processing image ", each # prints the name of image being processed

            # writes image name to csv
            textwriter.writerow([each])

            # reads image
            img = cv2.imread(each)

            # classifies image
            marker, midpoint = gd.find_faces(img, face_cascade, refObj, refObjMid, img_cent_orig)

            # sets up q iterator for multiple faces per image
            q = 0



            results.setdefault(each, [])
            # iterates through all observed faces in each image
            for every in marker:

                # gets observed face width in pixels from image for distance calculation
                faceWidth = gd.get_face_width(every)

                # finds distance between each face and camera and reference object
                inches_f, inches_r, feet_f, feet_r, distBetween = gd.get_distance(faceWidth, refWidth, angle, KNOWN_WIDTH_f, KNOWN_WIDTH_r, focalLength_f, focalLength_r)

                # triangulates face position relative to camera
                cam, ref, face, refTriangle, faceTriangle = gd.get_coord_pair(feet_f, midpoint[q], feet_r, img_center, height, refObjMid)

                # writes relative lat/lon of faces to lists for plotting
                faceX.append(int(faceTriangle[0]))
                faceY.append(int(faceTriangle[1]))

                # creates tuple of relative face positions per image
                latlon = (faceTriangle[0], faceTriangle[1])

                # appends positions to results dictionary
                results[each].append(latlon)

                # uses counter to scale colors
                j.append(i)
                q +=1

            # styles plot
           # plt.axis([-10, 10, 0, 30])
           # plt.set_cmap("Reds")

            # scales color ramp based on number of images. darker colors are newer observations
            numImg = len(images)
            c = [float(i) / float(numImg), 0.0, float(numImg - i) / float(numImg)]

            # creats scatter plot
            #plt.scatter(faceX, faceY, s=100, c=j, edgecolors='black')

            # adds grid to plot
            # plt.grid()

            # optionally saves plot to disk
            # plt.savefig("sheeplez{}.png".format(i))
            i += 1


            # adds text to image
            cv2.putText(img, "image:  %s" % (str(each)),
                        (img.shape[1] - 600, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 255, 0), 3)

            # resizes images so it's easier to see on screen, shows image
            img = imutils.resize(img, width=min(1000, img.shape[1]))
            cv2.imshow("image", img)

            # optionally saves image (with rectangles) to disk
            # cv2.imwrite("image{}.png".format(i), img)

            # the 1 below signifies that no user input is required to cycle through images
            # change to 0 if you want to be able to observe each image after classification
            cv2.waitKey(1)

            # shows matplotlib plot
            # plt.show()

            # closes current image and continues loop
            cv2.destroyAllWindows()

            print "-----------------------------"
    # writes results to json file
    json.dump(results, resOut)
