#!/usr/bin/python

import smach
import rospy
import cv2
import os
from collections import defaultdict
from operator import itemgetter

from euclidian_tracking import vector_euclidian_distance, EuclidianTracker
from math import sqrt

import numpy as np
from sensor_msgs.msg import Image
from robocup_face_recognition.srv import FaceDetection, GetEmbeddings, GetAgeAndGender


from cv_bridge import CvBridge, CvBridgeError


class Person:
    def __init__(self, face, name, top_colour, hair_colour, age, gender):
        self.face = face
        self.name = name
        self.top_colour = top_colour
        self.hair_colour = hair_colour
        self.age = age
        self.gender = gender


face_vectors = []
LAB_COLOUR_MAP = np.load(os.path.dirname(os.path.realpath(__file__)) + '/lab_colour_map.npy')
LAB_COLOURS = ['yellow', 'orange', 'red', 'magenta', 'navy', 'blue', 'teal', 'green', 'lime green']


# cosine similarity to measure distance between faces
def cosine_similarity(a,b):
    return np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b)

def colour_euclidian_distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += pow(a[i] - b[i],2)
    return sqrt(sum)

# get closest colour based on euclidian distance
def closest_colour(colour):
    l = colour[0]
    a = int((colour[1] + 128)/2)
    b = int((colour[2] + 128)/2)
    index = LAB_COLOUR_MAP[a,b]

    prefix = ''

    if l > 80:
        colour_name = 'white'
    elif l < 20:
        colour_name = 'black'
    else:
        if l < 40:
            prefix = 'dark '
        elif l > 60:
            prefix = 'light '
        if pow(a,2) + pow(b,2) < pow(90,2):
            colour_name = 'grey'
        else:
            colour_name = LAB_COLOURS[index]

    return prefix +  colour_name

def dict_most_common(count):
    max_key = -1
    max_count = 0
    for key in count:
        if count[key] > max_count:
            max_count = count[key]
            max_key = key
    return max_key

# create all of the clients
class Init(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        return 'outcome1'


# wait for someone to be in view for 10 seconds
class Waiting(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])
    
    def execute(self, userdata):
        rospy.wait_for_service('face_detection')
        get_faces = rospy.ServiceProxy('face_detection', FaceDetection)

        tracker = EuclidianTracker()
        bridge = CvBridge()
        
        while True:
            img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
            depth_msg = rospy.wait_for_message('/xtion/depth_registered/image_raw',Image)
            
            try:
                faces = get_faces(img_msg,0.5)
                frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
                frame_height, frame_width = frame.shape[:2]

                depth_frame = np.fromstring(depth_msg.data).view(dtype=np.float32).reshape(frame_height, frame_width)
                close_detections = []

                # mark faces within 1.5m as close
                for detection in faces.detections:
                    box = detection.box
                    face_depth = np.array(depth_frame[box[1] : box[3], box[0] : box[2]])
                    median = np.median(face_depth)
                    if median <= 1.5:
                        close_detections.append(detection)
                
                # update tracker for all close objects
                boxes = [detection.box for detection in close_detections]
                tracker.update(boxes)

                # output
                for tracked_object in tracker.tracked_objects:
                    box = tracked_object.box
                    cv2.putText(frame, 'id: {}'.format(tracked_object.id), (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                
                cv2.imshow('cosine_sim',frame)
                cv2.waitKey(1)
            
            except rospy.ServiceException as e:
                rospy.logerr(e)
            except CvBridgeError as e:
                rospy.logerr(e)
            
            # if a tracked person has been around for 10 seconds, transition
            max_time = 0
            for tracked_object in tracker.tracked_objects:
                time_elapsed = rospy.Time.now().secs - tracked_object.time.secs
                if time_elapsed >= 7:
                    return 'outcome1'
                if time_elapsed > max_time:
                    max_time = time_elapsed
            print max_time


# TODO: SELECT LARGEST & CENTREMOST FACE BOX
# TODO: (scale size by distance from centre to get this)
# TODO: perhaps look at the tracked person first as well
#
# identify features of said someone
class Detecting(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        rospy.wait_for_service('face_detection')
        rospy.wait_for_service('face_embedding')
        rospy.wait_for_service('face_age_and_gender')
        get_faces = rospy.ServiceProxy('face_detection', FaceDetection)
        get_embeddings = rospy.ServiceProxy('face_embedding', GetEmbeddings)
        get_age_and_gender = rospy.ServiceProxy('face_age_and_gender', GetAgeAndGender)
        bridge = CvBridge()

        face_vector_list = []
        face_dict = defaultdict(int)
        top_colour_dict = defaultdict(int)
        hair_colour_dict = defaultdict(int)
        age_dict = defaultdict(int)
        gender_dict = defaultdict(int)

        for iteration in range(10):
            img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
            top_colour = None
            hair_colour = None
            try:
                faces = get_faces(img_msg,0.5)
                frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
                frame_bb = bridge.imgmsg_to_cv2(faces.image_bb, "bgr8")
                frame_height, frame_width = frame.shape[:2]

                for detection in faces.detections:
                    box = detection.box
                    box_w = box[2] - box[0]
                    box_h = box[3] - box[1]



                    # COLOUR OF TOP
                    # body dims
                    body_x1 = max(box[0] - int(0.2 * box_w), 0)
                    body_y1 = min(box[3] + int(0.2 * box_h), frame_height)
                    body_x2 = min(box[2] + int(0.2 * box_w), frame_width)
                    body_y2 = min(box[3] + int(1.5 * box_h), frame_height)
                    body_box = (body_x1,body_y1,body_x2,body_y2)
                    # get all pixels of top
                    body_colours = frame[body_box[1] : body_box[3], body_box[0] : body_box[2]]
                    body_colours = cv2.cvtColor(body_colours, cv2.COLOR_BGR2LAB)
                    # k-means clustering
                    no_clusters = 3
                    if body_colours is not None and body_colours.size >= no_clusters:
                        body_colours = np.float32(body_colours.reshape(body_colours.size/3, 3))
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1) # used to be 200, 0.5
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        compactness, labels, centres = cv2.kmeans(body_colours,no_clusters,None,criteria,10,flags)
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
                        l,a,b = dominant_colour
                        l = l * 100/255
                        a = a - 128
                        b = b - 128
                        top_colour = closest_colour((l,a,b))

                        # pallette
                        centres_bgr = cv2.cvtColor(np.uint8([centres]), cv2.COLOR_LAB2BGR)[0]
                        colours_bgr = []
                        for i in range(len(centres_bgr)):
                            start_i = i * 40
                            colour = [int(i) for i in centres_bgr[i]]
                            colours_bgr.append(colour)
                            cv2.rectangle(frame_bb, (start_i, frame_height - 40), (start_i + 40, frame_height), colour, -1)

                        # output
                        cv2.putText(frame_bb, 'top colour: {}'.format(top_colour), (body_box[0], body_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        cv2.rectangle(frame_bb, (body_box[0], body_box[1]), (body_box[2], body_box[3]), colours_bgr[max_key], -1)
                        cv2.rectangle(frame_bb, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (0,0,255), 1)



                    # COLOUR OF HAIR
                    # hair dims
                    hair_x1 = max(box[0] - int(0.1 * box_w), 0)
                    hair_y1 = max(box[1] - int(0.1 * box_h), 0)
                    hair_x2 = min(box[2] + int(0.1 * box_w), frame_width)
                    hair_y2 = box[1] + int(0.3 * box_h)
                    hair_w_fifth = int(0.2 * box_w)
                    hair_h_fifth = int(0.2 * box_h)

                    # create a helmet to find colour
                    hair_colours_t = frame[hair_y1 : hair_y1 + hair_h_fifth, box[0] : box[2]]
                    hair_colours_l = frame[box[1] : hair_y2, hair_x1 : hair_x1 + hair_w_fifth]
                    hair_colours_r = frame[box[1] : hair_y2, hair_x2 - hair_w_fifth : hair_x2]

                    # concatenate all pixels to numpy array
                    hair_colours = hair_colours_t.reshape(hair_colours_t.size/3, 3)
                    hair_colours = np.concatenate((hair_colours, hair_colours_l.reshape(hair_colours_l.size/3, 3)))
                    hair_colours = np.concatenate((hair_colours, hair_colours_r.reshape(hair_colours_r.size/3, 3)))
                    hair_colours = hair_colours.reshape(1, hair_colours.size/3, 3)
                    hair_colours = cv2.cvtColor(hair_colours, cv2.COLOR_BGR2LAB)

                    # k-means clustering
                    no_clusters = 3
                    if hair_colours is not None and hair_colours.size >= no_clusters:
                        hair_colours = np.float32(hair_colours).reshape(hair_colours.size/3, 3)
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        compactness, labels, centres = cv2.kmeans(hair_colours,no_clusters,None,criteria,10,flags)
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
                        # get dominant colour in BGR, use RGB for colour name
                        dominant_colour = [int(i) for i in centres[max_key]]
                        l,a,b = dominant_colour
                        l = l * 100/255
                        a = a - 128
                        b = b - 128
                        hair_colour = closest_colour((l,a,b))

                        # pallette
                        centres_bgr = cv2.cvtColor(np.uint8([centres]), cv2.COLOR_LAB2BGR)[0]
                        colours_bgr = []
                        for i in range(len(centres_bgr)):
                            start_i = i * 40
                            colour = [int(i) for i in centres_bgr[i]]
                            colours_bgr.append(colour)
                            cv2.rectangle(frame_bb, (start_i, frame_height - 80), (start_i + 40, frame_height - 40), colour, -1)

                        # draw helmet
                        cv2.putText(frame_bb, 'hair colour: {}'.format(hair_colour), (box[0], box[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        cv2.rectangle(frame_bb, (box[0], hair_y1), (box[2], hair_y1 + hair_h_fifth), colours_bgr[max_key], -1)
                        cv2.rectangle(frame_bb, (hair_x1, box[1]), (hair_x1 + hair_w_fifth, hair_y2), colours_bgr[max_key], -1)
                        cv2.rectangle(frame_bb, (hair_x2 - hair_w_fifth, box[1]), (hair_x2, hair_y2), colours_bgr[max_key], -1)
                        cv2.rectangle(frame_bb, (box[0], hair_y1), (box[2], hair_y1 + hair_h_fifth), (0,0,255), 1)
                        cv2.rectangle(frame_bb, (hair_x1, box[1]), (hair_x1 + hair_w_fifth, hair_y2), (0,0,255), 1)
                        cv2.rectangle(frame_bb, (hair_x2 - hair_w_fifth, box[1]), (hair_x2, hair_y2), (0,0,255), 1)
                        


                    # AGE AND GENDER
                    # self-explanatory
                    face_x1 = max(0, int(box[0] - box_w/3))
                    face_y1 = max(0, int(box[1] - box_h/3))
                    face_x2 = min(frame_width, int(box[2] + box_w/3))
                    face_y2 = min(frame_height, int(box[3] + box_h/3))
                    face_box = (face_x1, face_y1, face_x2, face_y2)
                    age_and_gender = get_age_and_gender(img_msg, box)
                    age = age_and_gender.age
                    gender = age_and_gender.gender
                    cv2.putText(frame_bb, 'age: {}, gender: {}'.format(age,gender), (box[0], box[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.rectangle(frame_bb, (face_x1, face_y1), (face_x2, face_y2), (0,0,255), 1)



                    # FACE RECOGNITION
                    embeddings = get_embeddings(img_msg, box).embeddings
                    face_vector_list.append(embeddings)
                    match = False
                    # add new face if none
                    # if len(face_vectors) == 0:
                    #     face_vectors.append(embeddings)
                    # get max similarity
                    max_sim = -1
                    max_i = 0
                    for i in range(len(face_vectors)):
                        sim = cosine_similarity(face_vectors[i], embeddings)
                        if sim > max_sim:
                            max_sim = sim
                            max_i = i
                    # output if sufficient max similarity
                    if max_sim > 0.75:
                        face_dict[max_i] += 1
                        cv2.putText(frame_bb, 'match: {}, {:.3f}'.format('person ' + str(max_i),max_sim), (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)


                # append to lists
                if top_colour is not None:
                    top_colour_dict[top_colour] += 1
                if hair_colour is not None:
                    hair_colour_dict[hair_colour] += 1
                age_dict[age] += 1
                gender_dict[gender] += 1

                # display test
                cv2.imshow('cosine_sim',frame_bb)
                cv2.waitKey(1)

            except rospy.ServiceException as e:
                rospy.logerr(e)
            except CvBridgeError as e:
                rospy.logerr(e)
        
        matched_face_index = dict_most_common(face_dict)
        print matched_face_index
        if face_dict[matched_face_index] >= 3:
            matched_face = matched_face_index
        else:
            face_vectors.append(face_vector_list[0])
            matched_face = len(face_vectors) - 1

        print 'PERSON ESTIMATED:\n\
                top colour:  {}\n\
                hair colour: {}\n\
                age        : {}\n\
                gender     : {}\n\
                person     : {}'.format(dict_most_common(top_colour_dict),\
                                        dict_most_common(hair_colour_dict),\
                                        dict_most_common(age_dict),\
                                        dict_most_common(gender_dict),\
                                        matched_face)

        print 'waiting for human'
        return 'outcome1'



# DEPLOY THE AUTOMATON
def main():
    rospy.init_node('robocup_smach')
    sm = smach.StateMachine(outcomes=['outcome2'])

    with sm:
        # smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'WAITING'})
        smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'DETECTING'})
        smach.StateMachine.add('WAITING', Waiting(), transitions={'outcome1':'DETECTING'})
        smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'WAITING'})
    
    outcome = sm.execute()

if __name__ == '__main__':
    main()