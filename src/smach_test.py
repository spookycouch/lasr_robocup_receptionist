#!/usr/bin/python

import smach
import rospy
import cv2
from collections import defaultdict
from operator import itemgetter

from euclidian_tracking import vector_euclidian_distance, EuclidianTracker

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

def dict_most_common():
    pass

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

        while True:
            img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
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
                    body_colours = np.float32(body_colours.reshape(body_colours.size/3, 3))
                    # k-means clustering
                    no_clusters = 3
                    if body_colours.size >= no_clusters:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1) # used to be 200, 0.5
                        flags = cv2.KMEANS_RANDOM_CENTERS
                        compactness, labels, centres = cv2.kmeans(body_colours,no_clusters,None,criteria,10,flags)
                        # count max occurrences
                        count = defaultdict(int)
                        max_key = 0
                        max_count = 0
                        # TODO: create sorted list instead of dict
                        for label in labels:
                            count[label[0]] += 1
                        for key in count:
                            if count[key] > max_count:
                                max_count = count[key]
                                max_key = key
                        # get dominant colour in BGR, use RGB for colour name
                        dominant_colour = [int(i) for i in centres[max_key]]
                        colour_name = closest_colour(dominant_colour[::-1])

                        # output
                        cv2.putText(frame_bb, 'top colour: {}'.format(colour_name), (body_box[0], body_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        cv2.rectangle(frame_bb, (body_box[0], body_box[1]), (body_box[2], body_box[3]), dominant_colour, -1)
                        cv2.rectangle(frame_bb, (body_box[0], body_box[1]), (body_box[2], body_box[3]), (0,0,255), 1)
                        # palette
                        for i in range(len(centres)):
                            colour = [int(j) for j in centres[i]]
                            start_i = i * 40
                            cv2.rectangle(frame_bb, (start_i, frame_height - 40), (start_i + 40, frame_height), colour, -1)



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
                    hair_colours = np.float32(hair_colours)

                    # k-means clustering
                    no_clusters = 3
                    if hair_colours.size >= no_clusters:
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
                        colour_name = closest_colour(dominant_colour[::-1])

                        # draw helmet
                        cv2.putText(frame_bb, 'hair colour: {}'.format(colour_name), (box[0], box[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        cv2.rectangle(frame_bb, (box[0], hair_y1), (box[2], hair_y1 + hair_h_fifth), dominant_colour, -1)
                        cv2.rectangle(frame_bb, (hair_x1, box[1]), (hair_x1 + hair_w_fifth, hair_y2), dominant_colour, -1)
                        cv2.rectangle(frame_bb, (hair_x2 - hair_w_fifth, box[1]), (hair_x2, hair_y2), dominant_colour, -1)
                        cv2.rectangle(frame_bb, (box[0], hair_y1), (box[2], hair_y1 + hair_h_fifth), (0,0,255), 1)
                        cv2.rectangle(frame_bb, (hair_x1, box[1]), (hair_x1 + hair_w_fifth, hair_y2), (0,0,255), 1)
                        cv2.rectangle(frame_bb, (hair_x2 - hair_w_fifth, box[1]), (hair_x2, hair_y2), (0,0,255), 1)
                        # draw palette
                        for i in range(len(centres)):
                            colour = [int(j) for j in centres[i]]
                            start_i = i * 40
                            cv2.rectangle(frame_bb, (start_i, frame_height - 80), (start_i + 40, frame_height - 40), colour, -1)
                        


                    # AGE AND GENDER
                    # self-explanatory
                    age_and_gender = get_age_and_gender(img_msg, box)
                    age = age_and_gender.age
                    gender = age_and_gender.gender
                    cv2.putText(frame_bb, 'age: {}, gender: {}'.format(age,gender), (box[0], box[1] - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)



                    # FACE RECOGNITION
                    embeddings = get_embeddings(img_msg, box).embeddings
                    match = False
                    # add new face if none
                    if len(face_vectors) == 0:
                        face_vectors.append(embeddings)                    
                    # get max similarity
                    max_sim = -1
                    max_i = 0
                    for i in range(len(face_vectors)):
                        sim = cosine_similarity(face_vectors[i], embeddings)
                        if sim > max_sim:
                            max_sim = sim
                            max_i = i
                    # output if sufficient max similarity
                    if max_sim > 0.5:
                        cv2.putText(frame_bb, 'match: {}, {:.3f}'.format('person ' + str(max_i),max_sim), (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    else:
                        face_vectors.append(embeddings)


                # display test
                cv2.imshow('cosine_sim',frame_bb)
                cv2.waitKey(1)



            except rospy.ServiceException as e:
                rospy.logerr(e)
            except CvBridgeError as e:
                rospy.logerr(e)

        print 'waiting for human'
        return 'outcome1'



# DEPLOY THE AUTOMATON
def main():
    rospy.init_node('robocup_smach')
    sm = smach.StateMachine(outcomes=['outcome2'])

    with sm:
        smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'WAITING'})
        # smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'DETECTING'})
        smach.StateMachine.add('WAITING', Waiting(), transitions={'outcome1':'DETECTING'})
        smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'outcome2'})
    
    outcome = sm.execute()

if __name__ == '__main__':
    main()