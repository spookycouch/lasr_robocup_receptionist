#!/usr/bin/python

import cv2

from euclidian_tracking import vector_euclidian_distance, EuclidianTracker

import numpy as np




COLOURS = {
          'black'       : (0,0,0)       ,\
          'white'       : (255,255,255) ,\
          'red'         : (255,0,0)     ,\
          'light green' : (0,255,0)     ,\
          'blue'        : (0,0,255)     ,\
          'yellow'      : (255,255,0)   ,\
          'cyan'        : (0,255,255)   ,\
          'magenta'     : (255,0,255)   ,\
          'light grey'  : (192,192,192) ,\
          'grey'        : (128,128,128) ,\
          'maroon'      : (128,0,0)     ,\
          'olive'       : (128,128,0)   ,\
          'green'       : (0,128,0)     ,\
          'purple'      : (128,0,128)   ,\
          'teal'        : (0,128,128)   ,\
          'navy'        : (0,0,128)      \
          }


# cosine similarity to measure distance between faces
def cosine_similarity(a,b):
    return np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b)

# get closest colour based on euclidian distance
def closest_colour(colour):
    min_key = 'black'
    min_dist = vector_euclidian_distance(colour,COLOURS['black'])
    for key in COLOURS:
        curr_dist = vector_euclidian_distance(colour,COLOURS[key])
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_key = key
    return min_key


def gen_lab_spectrum():
    lab_grid_rgb = cv2.cvtColor(lab_grid, cv2.COLOR_LAB2BGR)
    lab_grid_rgb = np.multiply(lab_grid_rgb, 255)
    cv2.imwrite('lab_grid.png', lab_grid_rgb.astype(np.uint8))
    cv2.imshow('lab_grid',lab_grid_rgb.astype(np.uint8))
    cv2.waitKey(0)


# TODO:
# look into using CIE colour wheel for colours
def main():
    for key in COLOURS:
        colour_rgb = np.float32([[COLOURS[key]]])
        colour_lab = cv2.cvtColor(colour_rgb, cv2.COLOR_RGB2LAB).astype(np.int32)
        # print key, list(colour_lab[0][0])
    lab_grid = np.zeros((255,255,3), dtype = np.float32)
    print lab_grid.shape
    height, width = lab_grid.shape[0:2]
    print height,width
    for row in range(height):
        for col in range(width):
            lab_grid[row][col] = [75, col - 127, 127 - row]

    lab_grid_rgb = cv2.imread('lab_grid.png')
    lab_grid_rgb = lab_grid_rgb[:,:,0].astype(np.uint8)
    np.save('lab_colour_map.npy',lab_grid_rgb)
    print lab_grid_rgb[127,0]
    print lab_grid_rgb[127,127]


if __name__ == '__main__':
    main()