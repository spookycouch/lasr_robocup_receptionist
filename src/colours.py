#!/usr/bin/python
import cv2
import numpy as np
import os
from collections import defaultdict



LAB_COLOUR_MAP = np.load(os.path.dirname(os.path.realpath(__file__)) + '/lab_colour_map.npy')
LAB_COLOURS = ['yellow', 'orange', 'red', 'magenta', 'navy', 'blue', 'teal', 'green', 'lime green', 'grey']

# get closest colour based on euclidian distance
def closest_colour(colour):

    l,a,b = cv2.cvtColor(np.array([[colour]], np.uint8), cv2.COLOR_BGR2LAB)[0][0]
    l = l * 100/255
    a = int(a/2)
    b = 127 - int(b/2)

    index = LAB_COLOUR_MAP[b,a]

    prefix = ''
    colour_name = 'grey'

    # colour is not grey if at least one pair of values differs by 10
    for i in range(len(colour)):
        for j in range(len(colour))[1:]:
            if abs(colour[i] - colour[j]) > 10:
                colour_name = LAB_COLOURS[index]
        
    if l < 30:
        prefix = 'dark '
    elif l > 70:
        prefix = 'light '

    # special cases for black and white
    if colour_name == 'grey':
        if l < 15:
            return 'black'
        elif l > 85:
            return 'white'
    elif l < 10:
        return 'black'
    elif l > 90:
        return 'white'

    return prefix +  colour_name



# k-means clustering
def k_means_colour(no_clusters, pixels):
    if pixels.size >= no_clusters:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1) # used to be 200, 0.5
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centres = cv2.kmeans(pixels,no_clusters,None,criteria,10,flags)
        # count max occurrences
        count = defaultdict(int)
        max_key = 0
        max_count = 0
        for label in labels:
            count[label[0]] += 1
        for key in count:
            if count[key] > max_count:
                max_count = count[key]
                max_key = key
        # get dominant colour in LAB for colour name
        dominant_colour = [int(i) for i in centres[max_key]]
        colour_name = closest_colour(dominant_colour)
        return dominant_colour, colour_name, centres
    
    return (None, None, None)