#!/usr/bin/python

import smach
import rospy
import cv2
import actionlib
from math import sqrt, atan2
from cv_bridge import CvBridge, CvBridgeError
from operator import itemgetter

from vision import identify_person, wait_for_person
from euclidian_tracking import vector_euclidian_distance, EuclidianTracker

import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from pal_interaction_msgs.msg import TtsAction, TtsGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from control_msgs.msg import PointHeadAction, PointHeadGoal
from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, Quaternion, PointStamped, Vector3, PoseWithCovarianceStamped
from lasr_object_detection_yolo.srv import YoloDetection
import tf

from face_detector_tracker import FaceRecogniser



class GlobalState:
    def __init__(self):
        self.persons = []
        self.NAMES_M = ['bobbert','michael','steve']
        self.NAMES_F = ['sally','christine','jenny']
        self.tracker = EuclidianTracker()
        self.tracked_id = 0
        self.next_person_id = 0
        self.curr_person_id = None
        self.face_recogniser = FaceRecogniser(face_recog_confidence=0.65)
        self.target_chair = None



class Person:
    def __init__(self, id, name, top_colour, hair_colour, age, gender):
        self.id = id
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


def get_angle_between_points(source_x, source_y, target_x, target_y):
    dist_x = target_x - source_x
    dist_y = target_y - source_y
    return atan2(dist_y, dist_x)


def get_centre_point(x1, y1, x2, y2):
    return (x1 + x2)/2, (y1 + y2)/2


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
        world.tracked_id = wait_for_person(world.tracker)
        return 'outcome1'


# identify features of said someone
class Detecting(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1','wait_leave'])

    def execute(self, userdata):
        talk('Hi there, may I please look into your eyes?')
        result = identify_person(world.tracker, world.tracked_id, world.face_recogniser)
        success, recognised, person_id, top_colour_out, hair_colour_out, age_out, gender_out = result

        if not success:
            return 'outcome1'

        if recognised:
            person = world.persons[person_id]
            talk('Welcome back {}, I like your {} top!'.format(person.name, person.top_colour))
        else:
            name_out = np.random.choice(world.NAMES_M)
            person = Person(person_id, name_out, top_colour_out, hair_colour_out, age_out, gender_out)
            world.persons.append(person)
            talk('It appears I haven\'t met you before, I shall call you {}'.format(person.name))
        
        world.curr_person_id = person.id
        print person
        print len(world.persons)
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
        # TODO: for chair in chairs, look at point then do detection
        point_head_client = actionlib.SimpleActionClient('/head_controller/point_head_action', PointHeadAction)
        transformer = tf.TransformListener()

        chairs = rospy.get_param('/chairs')
        for chair in chairs:
            max_x, max_y = chairs[chair]['box']['max_xy']
            min_x, min_y = chairs[chair]['box']['min_xy']
            rospy.set_param('/chairs/' + chair + '/status', 'free')

            point_head_client.wait_for_server()
            chair_point = Point(max_x, max_y, 1)
            
            # create head goal
            ph_goal = PointHeadGoal()
            ph_goal.target.header.frame_id = 'map'
            ph_goal.max_velocity = 1
            ph_goal.min_duration = rospy.Duration(0.5)
            ph_goal.target.header.stamp = rospy.Time(0)
            ph_goal.target.point = chair_point
            ph_goal.pointing_frame = 'head_2_link'
            ph_goal.pointing_axis = Vector3(1,0,0)

            point_head_client.send_goal(ph_goal)
            point_head_client.wait_for_result()
            
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

            for person_point in person_coords:
                point = person_point.point
                if point.x > min_x and point.x < max_x and point.y > min_y and point.y < max_y:
                    rospy.set_param('/chairs/' + chair + '/status', 'taken')


            frame = bridge.imgmsg_to_cv2(detection_result.image_bb, "bgr8")
            cv2.imshow('frame', frame)
            cv2.waitKey(3)
            cv2.waitKey(3)

        return 'outcome1'

class GoToChair(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1', 'outcome2'])

    def execute(self, userdata):
        move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        move_base_client.wait_for_server()
        
        chairs = rospy.get_param('/chairs')
        for chair in chairs:
            if chairs[chair]['status'] == 'free':
                # get points and distances
                amcl_msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
                robot_point = amcl_msg.pose.pose.position
    
                chair_min_x, chair_min_y = chairs[chair]['box']['min_xy']
                chair_max_x, chair_max_y = chairs[chair]['box']['max_xy']
                chair_x, chair_y = get_centre_point(chair_min_x, chair_min_y, chair_max_x, chair_max_y)

                dist_x = chair_x - robot_point.x
                dist_y = chair_y - robot_point.y
                euclidian_dist = sqrt(dist_x * dist_x + dist_y * dist_y)
                
                target_dist = 1.5

                # calculate target point if euclidian distance is not within threshold.
                # otherwise tiago is nearby, rotate around current point.
                if euclidian_dist > target_dist + (target_dist/10):
                    # ratio of (desired dist)/(total dist)
                    ratio = (euclidian_dist - target_dist)/euclidian_dist
                    # add (ratio * actual dist) to robot point, basically scale the triangle
                    target_x = robot_point.x + (ratio * dist_x)
                    target_y = robot_point.y + (ratio * dist_y)
                else:
                    target_x = robot_point.x
                    target_y = robot_point.y
    
                target_point = Point(target_x, target_y, 0)

                # get new rotation
                target_angle = get_angle_between_points(target_x, target_y, chair_x, chair_y)
                (x, y, z, w) = tf.transformations.quaternion_from_euler(0, 0, target_angle)
                target_quaternion = Quaternion(x, y, z, w)
    
                # create and send move base goal
                mb_goal = MoveBaseGoal()
                mb_goal.target_pose.header.frame_id = 'map'
                mb_goal.target_pose.header.stamp = rospy.Time.now()
                mb_goal.target_pose.pose.position = target_point
                mb_goal.target_pose.pose.orientation = target_quaternion
    
                move_base_client.send_goal(mb_goal)
                print 'going to free chair'

                if move_base_client.wait_for_result():
                    rospy.loginfo('Goal location achieved!')

                    talk('please take a seat')

                    # check neighbours
                    for neighbour in chairs[chair]['neighbours']:
                        if chairs[neighbour]['status'] == 'taken':
                            world.target_chair = neighbour
                            return 'outcome2'
                else:
                    rospy.logwarn("Couldn't reach the goal!")
                
                return 'outcome1'

        talk('sorry, we are full. please go away')
        return 'outcome1'

class Introduce(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        if world.target_chair is not None:
            move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
            move_base_client.wait_for_server()

            chairs = rospy.get_param('/chairs')

            amcl_msg = rospy.wait_for_message('/amcl_pose', PoseWithCovarianceStamped)
            robot_point = amcl_msg.pose.pose.position

            chair_max_x, chair_max_y = chairs[world.target_chair]['box']['max_xy']
            chair_min_x, chair_min_y = chairs[world.target_chair]['box']['min_xy']
            chair_x, chair_y = get_centre_point(chair_min_x, chair_min_y, chair_max_x, chair_max_y)

            target_angle = get_angle_between_points(robot_point.x, robot_point.y, chair_x, chair_y)
            (x, y, z, w) = tf.transformations.quaternion_from_euler(0, 0, target_angle)
            target_quaternion = Quaternion(x, y, z, w)

            mb_goal = MoveBaseGoal()
            mb_goal.target_pose.header.frame_id = 'map'
            mb_goal.target_pose.header.stamp = rospy.Time.now()
            mb_goal.target_pose.pose.position = robot_point
            mb_goal.target_pose.pose.orientation = target_quaternion
    
            move_base_client.send_goal(mb_goal)

            if move_base_client.wait_for_result():
                rospy.loginfo('Goal location achieved!')
            else:
                rospy.logwarn("Couldn't reach the goal!")

            # try to identify the next seated person
            tries = 0
            recognised = False
            while not recognised and tries < 2: 
                world.tracked_id = wait_for_person(world.tracker, timeout=5)
                if world.tracked_id is not None:
                    result = identify_person(world.tracker, world.tracked_id, world.face_recogniser, add_mode=False)

                    if result[1]:
                        person = world.persons[result[2]]
                        talk('have you met {}'.format(person.name))
                        return 'outcome1'
                
                tries += 1
            
            talk('huh, looks like they left')

        return 'outcome1'


class GoHome(smach.State):
    def __init__(self):
        smach.State.__init__(self,outcomes=['outcome1'])

    def execute(self, userdata):
        move_base_client = actionlib.SimpleActionClient('/move_base', MoveBaseAction)
        move_base_client.wait_for_server()

        location = rospy.get_param('/home')

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




# DEPLOY THE AUTOMATON
def main():
    rospy.init_node('robocup_smach')
    sm = smach.StateMachine(outcomes=['outcome2'])

    with sm:
        smach.StateMachine.add('INIT', Init(), transitions={'outcome1':'GO_HOME'})
        smach.StateMachine.add('GO_HOME', GoHome(), transitions={'outcome1':'WAIT_NEW'})
        smach.StateMachine.add('WAIT_NEW', Waiting(), transitions={'outcome1':'DETECTING'})
        # smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'WAIT_NEW', 'wait_leave':'WAIT_LEAVE'})
        # smach.StateMachine.add('WAIT_LEAVE', WaitLeave(), transitions={'outcome1':'WAIT_NEW'})
        # vvv TEST THE DETECTION ISNT BROKE
        # smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'WAIT_NEW', 'wait_leave':'WAIT_NEW'})
        smach.StateMachine.add('DETECTING', Detecting(), transitions={'outcome1':'WAIT_NEW', 'wait_leave':'INSPECT_ROOM'})
        smach.StateMachine.add('WAIT_LEAVE', WaitLeave(), transitions={'outcome1':'INSPECT_ROOM'})
        smach.StateMachine.add('INSPECT_ROOM', InspectRoom(), transitions={'outcome1':'DETECT_PEOPLE'})
        smach.StateMachine.add('DETECT_PEOPLE', DetectPeople(), transitions={'outcome1':'GO_TO_CHAIR'})
        smach.StateMachine.add('GO_TO_CHAIR', GoToChair(), transitions={'outcome1':'GO_HOME', 'outcome2':'INTRODUCE'})
        smach.StateMachine.add('INTRODUCE', Introduce(), transitions={'outcome1':'GO_HOME'})

    
    outcome = sm.execute()

if __name__ == '__main__':
    main()