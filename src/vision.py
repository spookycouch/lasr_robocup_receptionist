import rospy
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from collections import defaultdict

from colours import k_means_colour

from sensor_msgs.msg import Image
from robocup_face_recognition.srv import FaceDetection, GetEmbeddings, GetAgeAndGender

def dict_most_common(count):
    max_key = -1
    max_count = 0
    for key in count:
        if count[key] > max_count:
            max_count = count[key]
            max_key = key
    return max_key

def update_tracker_faces_and_depth(tracker, faces, depth_msg):
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
    tracker.update(boxes)


def identify_person(tracker, tracked_id, face_recogniser, add_mode=True):
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

            update_tracker_faces_and_depth(tracker, faces, depth_msg)

            # if the tracked face is no longer present, return to wait
            detection_focus = None
            for detection in tracker.tracked_objects:
                if detection.id == tracked_id:
                    detection_focus = detection
            if detection_focus is None:
                return 0, None, None, None, None, None, None

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
            max_id, max_sim = face_recogniser.recogniseFaceFromEmbedding([embeddings])
            
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
            print 'tracking {}: {:.2f} match person {}'.format(tracked_id, max_sim, max_id)
            cv2.imshow('cosine_sim',frame_bb)
            cv2.waitKey(1)

        except rospy.ServiceException as e:
            rospy.logerr(e)
        except CvBridgeError as e:
            rospy.logerr(e)
    

    top_colour_out = dict_most_common(top_colour_dict)
    hair_colour_out = dict_most_common(hair_colour_dict)
    age_out = dict_most_common(age_dict)
    gender_out = dict_most_common(gender_dict)

    person = None
    matched_face_index = None
    matched_face = dict_most_common(face_dict)
    
    try:
        matched_face_index = int(matched_face)
    except ValueError:
        pass
    
    print face_dict

    if face_dict[matched_face] >= 5 and matched_face_index is not None:
        return 1, 1, matched_face_index, top_colour_out, hair_colour_out, age_out, gender_out
    elif add_mode:
        # id is order added to SVM, -1 accounting for 'unknown' class
        person_id = len(face_recogniser.get_classnames()) - 1

        face_recogniser.add_embeddings_to_temp(face_vector_list, str(person_id))
        face_recogniser.train_classifier()
        
        print 'person {} added to SVM'.format(person_id)
        return 1, 0, matched_face_index, top_colour_out, hair_colour_out, age_out, gender_out
    else:
        return 1, 0, None, None, None, None, None



def wait_for_person(tracker, timeout=None):
    rospy.wait_for_service('face_detection')
    get_faces = rospy.ServiceProxy('face_detection', FaceDetection)
    bridge = CvBridge()
    
    start_time = rospy.Time.now()

    # reset time
    for detection in tracker.tracked_objects:
        detection.time = rospy.Time.now()
    
    while True:
        img_msg = rospy.wait_for_message('/xtion/rgb/image_raw',Image)
        depth_msg = rospy.wait_for_message('/xtion/depth_registered/image_raw',Image)
        
        try:
            faces = get_faces(img_msg,0.5)
            update_tracker_faces_and_depth(tracker, faces, depth_msg)
            frame = bridge.imgmsg_to_cv2(img_msg, "bgr8")

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
        max_id = None
        for tracked_object in tracker.tracked_objects:
            time_elapsed = rospy.Time.now().secs - tracked_object.time.secs
            if time_elapsed > max_time:
                max_time = time_elapsed
                max_id = tracked_object.id
        
        print 'tracking {}: {} seconds'.format(max_id, max_time)

        if max_time >= 4:
            return max_id
        
        if timeout is not None:
            if rospy.Time.now().secs - start_time.secs > timeout:
                return None