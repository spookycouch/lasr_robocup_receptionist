import sys
import networkx as nx
import rospy
from math import sqrt


def vector_euclidian_distance(a,b):
    dist = 0
    for i in range(len(a)):
        dist += pow(a[i] - b[i], 2)
    return sqrt(dist)


def box_centre_point(box):
    return [int(box[0] + box[2]/2), int(box[1] + box[3]/2)]


class TrackedObject():
    def __init__(self, id, box, time):
        self.id = id
        self.box = box
        self.time = time


# Track objects on euclidian distance
class EuclidianTracker():
    def __init__(self):
        self.id = 0
        self.tracked_objects = []
    
    def update(self, boxes):
        redetected_boxes = [0] * len(boxes)
        redetected_objects = [0] * len(self.tracked_objects)

        if len(boxes) > 0 and len(self.tracked_objects) > 0:
            # create complete bipartite graph G of boxes to tracked objects,
            # where edge weight is (maxint - euclidian_distance),
            # such that max-matching can be performed to find closest matches
            G = nx.Graph()
            for i in range(len(boxes)):
                for j in range(len(self.tracked_objects)):
                    tracked_box = self.tracked_objects[j].box
                    euclid_dist = vector_euclidian_distance(box_centre_point(boxes[i]), box_centre_point(tracked_box))
                    weight = sys.maxint - euclid_dist

                    # set weight = 0 if box is more than one box away (discard)
                    box_dist = vector_euclidian_distance((tracked_box[0],tracked_box[1]), (tracked_box[2],tracked_box[3]))
                    if euclid_dist > box_dist:
                        weight = 0
                    G.add_edge(('b', i), ('t', j), weight = weight)
            
            # max-match using Blossom algorithm
            max_match = nx.max_weight_matching(G)
            for match in max_match:
                if not G[('b', match[0][1])][('t',match[1][1])]['weight'] == 0:
                    self.tracked_objects[match[1][1]].box = boxes[match[0][1]]
                    redetected_boxes[match[0][1]] = 1
                    redetected_objects[match[1][1]] = 1

        # delete dropped objects        
        for i in range(len(redetected_objects)):
            if redetected_objects[i] == 0:
                del self.tracked_objects[i]
        
        # add new detections
        for i in range(len(redetected_boxes)):
            if redetected_boxes[i] == 0:
                self.tracked_objects.append(TrackedObject(self.id, boxes[i], rospy.Time.now()))
                self.id += 1
        