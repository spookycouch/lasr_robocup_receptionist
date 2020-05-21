#!/usr/bin/python

import smach
import rospy
import cv2
import actionlib
from cv_bridge import CvBridge, CvBridgeError
from collections import defaultdict
from operator import itemgetter

from euclidian_tracking import vector_euclidian_distance, EuclidianTracker
from colours import k_means_colour

import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, Quaternion, PointStamped
from lasr_object_detection_yolo.srv import YoloDetection
from robocup_face_recognition.srv import FaceDetection, GetEmbeddings, GetAgeAndGender
import tf



class GlobalState:
    def __init__(self):
        self.persons = []
        self.NAMES_M = ['bobbert','michael','steve']
        self.NAMES_F = ['sally','christine','jenny']
        self.tracker = EuclidianTracker()
        self.tracked_id = 0
        self.next_person_id = 0
        self.curr_person_id = None



class Person:
    def __init__(self, id, face, name, top_colour, hair_colour, age, gender):
        self.id = id
        self.face = face
        self.name = name
        self.top_colour = top_colour
        self.hair_colour = hair_colour
        self.age = age
        self.gender = gender
    
    def __str__(self):
        return 'PERSON {}: \
            \n\tname       : {}\
            \n\ttop colour : {}\
            \n\thair colour: {}\
            \n\tage        : {}\
            \n\tgender     : {}'.format(self.id,        \
                                    self.name,          \
                                    self.top_colour,    \
                                    self.hair_colour,   \
                                    self.age,           \
                                    self.gender)



world = GlobalState()



def talk(text):
    # Create the TTS goal and send it
    speech_client = actionlib.SimpleActionClient('/tts', TtsAction)
    print('\033[1;36mTIAGO: ' + text + '\033[0m')
    tts_goal = TtsGoal()
    tts_goal.rawtext.lang_id = 'en_GB'
    tts_goal.rawtext.text = text
    rospy.sleep(0.3) # prevent race conditions
    speech_client.send_goal(tts_goal)



# cosine similarity to measure distance between faces
def cosine_similarity(a,b):
    return np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b)
    


def dict_most_common(count):
    max_key = -1
    max_count = 0
    for key in count:
        if count[key] > max_count:
            max_count = count[key]
            max_key = key
    return max_key



def update_tracker_faces_and_depth(faces, depth_msg):
    depth_frame = np.fromstring(depth_msg.data).view(dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
    close_detections = []

    # mark faces within 1.5m as close
    for detection in faces.detections:
        box = detection.box
        face_depth = np.array(depth_frame[box[1] : box[3], box[0] : box[2]])
        face_depth = face_depth[np.isfinite(face_depth)]
        median = np.median(face_depth)
        if median <= 2:
            close_detections.append(detection)
        # close_detections.append(detection)
    
    # update tracker for all close objects
    boxes = [detection.box for detection in close_detections]
    world.tracker.update(boxes)



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
        bridge = CvBridge()

        # reset time
        for detection in world.tracker.tracked_objects:
            detection.time = rospy.Time.now()
        
        while True:
            img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
            depth_msg = rospy.wait_for_message('/xtion/depth_registered/image_raw',Image)
            
            try:
                faces = get_faces(img_msg,0.5)
                update_tracker_faces_and_depth(faces, depth_msg)
                frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")

                # output
                for tracked_object in world.tracker.tracked_objects:
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
            max_id = 0
            for tracked_object in world.tracker.tracked_objects:
                time_elapsed = rospy.Time.now().secs - tracked_object.time.secs
                if time_elapsed > max_time:
                    max_time = time_elapsed
                    max_id = tracked_object.id
            
            print 'tracking {}: {} seconds'.format(max_id, max_time)

            if max_time >= 4:
                world.tracked_id = max_id
                return 'outcome1'


# identify features of said someone
class Detecting(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1','wait_leave'])

    def execute(self, userdata):
        talk('Hi there, may I please look into your eyes?')
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
            depth_msg = rospy.wait_for_message('/xtion/depth_registered/image_raw',Image)
            top_colour = None
            hair_colour = None
            try:
                faces = get_faces(img_msg,0.5)
                frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")
                frame_bb = bridge.imgmsg_to_cv2(faces.image_bb, "bgr8")
                frame_height, frame_width = frame.shape[:2]

                update_tracker_faces_and_depth(faces, depth_msg)

                # if the tracked face is no longer present, return to wait
                detection_focus = None
                for detection in world.tracker.tracked_objects:
                    if detection.id == world.tracked_id:
                        detection_focus = detection
                if detection_focus is None:
                    return 'outcome1'

                box = detection_focus.box
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
                # get dominant colour in LAB for colour name
                dominant_colour, top_colour, centres = k_means_colour(3, body_colours)

                if dominant_colour is not None:
                    # pallette
                    colours_bgr = []
                    for i in range(len(centres)):
                        start_i = i * 40
                        colour = [int(i) for i in centres[i]]
                        colours_bgr.append(colour)
                        cv2.rectangle(frame_bb, (start_i, frame_height - 40), (start_i + 40, frame_height), colour, -1)

                    # output
                    cv2.putText(frame_bb, 'top colour: {}'.format(top_colour), (body_box[0], body_box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.rectangle(frame_bb, (body_box[0], body_box[1]), (body_box[2], body_box[3]), dominant_colour, -1)
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
                hair_colours = np.float32(hair_colours).reshape(hair_colours.size/3, 3)


                # k-means clustering
                dominant_colour, hair_colour,centres = k_means_colour(3,hair_colours)
            
                if dominant_colour is not None:

                    # pallette
                    colours_bgr = []
                    for i in range(len(centres)):
                        start_i = i * 40
                        colour = [int(i) for i in centres[i]]
                        colours_bgr.append(colour)
                        cv2.rectangle(frame_bb, (start_i, frame_height - 80), (start_i + 40, frame_height - 40), colour, -1)

                    # draw helmet
                    cv2.putText(frame_bb, 'hair colour: {}'.format(hair_colour), (box[0], box[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.rectangle(frame_bb, (box[0], hair_y1), (box[2], hair_y1 + hair_h_fifth), dominant_colour, -1)
                    cv2.rectangle(frame_bb, (hair_x1, box[1]), (hair_x1 + hair_w_fifth, hair_y2), dominant_colour, -1)
                    cv2.rectangle(frame_bb, (hair_x2 - hair_w_fifth, box[1]), (hair_x2, hair_y2), dominant_colour, -1)
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
                # get max similarity
                max_sim = -1
                max_id = None
                for person in world.persons:
                    sim = cosine_similarity(person.face, embeddings)
                    if sim > max_sim:
                        max_sim = sim
                        max_id = person.id
                # output if sufficient max similarity
                cv2.putText(frame_bb, 'match: {}, {:.3f}'.format('person ' + str(max_id),max_sim), (box[0], box[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                if max_sim > 0.7:
                    face_dict[max_id] += 1


                # append to lists
                if top_colour is not None:
                    top_colour_dict[top_colour] += 1
                if hair_colour is not None:
                    hair_colour_dict[hair_colour] += 1
                age_dict[age] += 1
                gender_dict[gender] += 1

                # display test
                print 'tracking {}: {:.2f} match person {}'.format(world.tracked_id, max_sim, max_id)
                cv2.imshow('cosine_sim',frame_bb)
                cv2.waitKey(1)

            except rospy.ServiceException as e:
                rospy.logerr(e)
            except CvBridgeError as e:
                rospy.logerr(e)
        

        name_out = np.random.choice(world.NAMES_M)
        top_colour_out = dict_most_common(top_colour_dict)
        hair_colour_out = dict_most_common(hair_colour_dict)
        age_out = dict_most_common(age_dict)
        gender_out = dict_most_common(gender_dict)

        person = None
        matched_face_index = dict_most_common(face_dict)
        print face_dict, len(world.persons)

        if face_dict[matched_face_index] >= 5:
            person = world.persons[matched_face_index]
            talk('Welcome back {}, I like your {} top!'.format(person.name, person.top_colour))
        else:
            # get closest to average face vector and append it
            average_face = np.average(face_vector_list,axis=0)
            face_dist_avg = []
            for face_vector in face_vector_list:
                face_dist_avg.append(vector_euclidian_distance(face_vector, average_face))
            face_index = np.argmin(face_dist_avg)

            person = Person(world.next_person_id, face_vector_list[face_index], name_out, top_colour_out, hair_colour_out, age_out, gender_out)
            world.persons.append(person)
            world.next_person_id += 1
            talk('It appears I haven\'t met you before, I shall call you {}'.format(person.name))
        
        world.curr_person_id = person.id
        print person
        return 'wait_leave'



class WaitLeave(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        rospy.wait_for_service('face_detection')
        get_faces = rospy.ServiceProxy('face_detection', FaceDetection)
        bridge = CvBridge()

        start_time = rospy.Time.now()
        spoken = False
        
        while True:
            img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
            depth_msg = rospy.wait_for_message('/xtion/depth_registered/image_raw',Image)
            
            try:
                faces = get_faces(img_msg,0.5)
                update_tracker_faces_and_depth(faces, depth_msg)
                frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")

                # output
                for tracked_object in world.tracker.tracked_objects:
                    box = tracked_object.box
                    cv2.putText(frame, 'id: {}'.format(tracked_object.id), (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
                
                cv2.imshow('cosine_sim',frame)
                cv2.waitKey(1)
            
            except rospy.ServiceException as e:
                rospy.logerr(e)
            except CvBridgeError as e:
                rospy.logerr(e)
            
            time_elapsed = None

            for tracked_object in world.tracker.tracked_objects:
                if tracked_object.id == world.tracked_id:
                    time_elapsed = rospy.Time.now().secs - start_time.secs

            if time_elapsed is not None:
                if not spoken and time_elapsed > 10:
                    talk('please begone {}, I would like to meet someone new'.format(world.persons[world.curr_person_id].name))
                    spoken = True
                print 'tracking {}: {} seconds'.format(world.tracked_id, time_elapsed)
            else:
                talk('goodbye {}'.format(world.persons[world.curr_person_id].name))
                return 'outcome1'



class InspectRoom(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        move_base_client.wait_for_server()

        location = rospy.get_param('/location')

        goal = MoveBaseGoal()
        goal.target_pose.header = Header(frame_id="map", stamp=rospy.Time.now())
        goal.target_pose.pose = Pose(position = Point(**location['position']),
            orientation = Quaternion(**location['orientation']))
        
        rospy.loginfo('Sending goal location ...')
        move_base_client.send_goal(goal) 
        if move_base_client.wait_for_result():
            rospy.loginfo('Goal location achieved!')
        else:
            rospy.logwarn("Couldn't reach the goal!")
        
        return 'outcome1'



class DetectPeople(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        # wait for the service to come up
        image_raw = rospy.wait_for_message('/xtion/rgb/image_raw', Image)
        rospy.wait_for_service('/yolo_detection')
        bridge = CvBridge()

        # call the service
        try:
            detect_objects = rospy.ServiceProxy('/yolo_detection', YoloDetection)
            detection_result = detect_objects(image_raw, 'coco', 0.5, 0.3)
        except rospy.ServiceException as e:
            print "Service call failed: %s"%e
            return 'outcome1'
        
        person_coords = []



        # -------------------------- #
        # --- WAIT FOR TRANSFORM --- #
        # -------------------------- #
        transformer = tf.TransformListener()
        rospy.sleep(2.0)

        depth_points = rospy.wait_for_message('xtion/depth_registered/points', PointCloud2)
        header = depth_points.header
        height = depth_points.height
        width = depth_points.width
        cloud = np.fromstring(depth_points.data, np.float32)
        cloud = cloud.reshape(height, width, 8)
        transformer.waitForTransform('xtion_rgb_optical_frame', 'map', depth_points.header.stamp, rospy.Duration(2.0))

        for person in detection_result.detected_objects:
            if not person.name == 'person':
                continue
            
            region_size = 2
            while True:
                # calculate centre points
                centre_x = int((person.xywh[0] + person.xywh[2]/2) - region_size)
                centre_y = int((person.xywh[1] + person.xywh[3]/2) - region_size)
                # extract xyz values along points of interest
                centre_cluster = cloud[centre_y  : centre_y + region_size, centre_x : centre_x + region_size, 0:3]
                not_nan_count = 0

                for axes in centre_cluster:
                    for point in axes:
                        if not (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2])):
                            not_nan_count += 1

                if not_nan_count >= 3:
                    break
                
                region_size += 2

            mean = np.nanmean(centre_cluster, axis=1)
            mean = np.nanmean(mean, axis=0)
            centre_point = PointStamped()
            centre_point.header = depth_points.header
            centre_point.point = Point(*mean)

            person_point = transformer.transformPoint('map', centre_point)
            print person_point
            person_coords.append(person_point)
        # -------------------------------------- #
        # --- WE SHOULD GET RID OF THIS SOON --- #
        # -------------------------------------- #



        chairs = rospy.get_param('/chairs')
        for chair in chairs:
            max_x, max_y = chairs[chair]['box']['max_xy']
            min_x, min_y = chairs[chair]['box']['min_xy']
            rospy.set_param('/chairs/' + chair + '/status', 'free')
            for person_point in person_coords:
                point = person_point.point
                if point.x > min_x and point.x < max_x and point.y > min_y and point.y < max_y:
                    rospy.set_param('/chairs/' + chair + '/status', 'taken')


        frame = bridge.imgmsg_to_cv2(detection_result.image_bb, "bgr8")
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

        return 'outcome1'








# DEPLOY THE AUTOMATON
def main():
    rospy.init_node('robocup_smach')
    sm = smach.StateMachine(outcomes=['outcome2'])

    with sm:
        smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'WAIT_NEW'})
        smach.StateMachine.add('WAIT_NEW', Waiting(), transitions={'outcome1':'DETECTING'})
        smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'WAIT_NEW', 'wait_leave':'INSPECT_ROOM'})
        smach.StateMachine.add('WAIT_LEAVE', WaitLeave(), transitions={'outcome1':'INSPECT_ROOM'})
        smach.StateMachine.add('INSPECT_ROOM', InspectRoom(), transitions={'outcome1':'DETECT_PEOPLE'})
        smach.StateMachine.add('DETECT_PEOPLE', DetectPeople(), transitions={'outcome1':'WAIT_NEW'})

    
    outcome = sm.execute()

if __name__ == '__main__':
    main()