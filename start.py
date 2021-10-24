import argparse
import os
from time import time
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import align.detect_face as detect_face
import cv2
import numpy as np
import tensorflow as tf
from lib.face_utils import judge_side_face
from lib.utils import Logger, mkdir
from project_root_dir import project_dir
from src.sort import Sort
import csv
import json
import pandas as pd
logger = Logger()

def main():
    global colours, img_size
    args = parse_args()
    videos_dir = args.videos_dir
    output_path = args.output_path
    no_display = args.no_display
    detect_interval = args.detect_interval  # you need to keep a balance between performance and fluency
    margin = args.margin  # if the face is big in your video ,you can set it bigger for tracking easiler
    scale_rate = args.scale_rate  # if set it smaller will make input frames smaller
    show_rate = args.show_rate  # if set it smaller will dispaly smaller frames
    face_score_threshold = args.face_score_threshold

    mkdir(output_path)
    # for display
    if not no_display:
        colours = np.random.rand(32, 3)

    # init tracker
    tracker = Sort()  # create instance of the SORT tracker

    logger.info('Start track and extract......')
    with tf.Graph().as_default():
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
                                              log_device_placement=False)) as sess:
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(project_dir, "align"))

            minsize = 10  # minimum size of face for mtcnn to detect
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor

            for filename in os.listdir(videos_dir):
                logger.info('All files:{}'.format(filename))
            for filename in os.listdir(videos_dir):
                suffix = filename.split('.')[1]
                if suffix != 'mp4' and suffix != 'avi' and suffix != 'webm':  # you can specify more video formats if you need
                    continue
                video_name = os.path.join(videos_dir, filename)
                directoryname = os.path.join(output_path, filename.split('.')[0])
                logger.info('Video_name:{}'.format(video_name))
                cam = cv2.VideoCapture(video_name)
                # Finding frames per second for the inputted video
                fps = cam.get(cv2.CAP_PROP_FPS)

                # cv2.namedWindow('frame',0)
                # cv2.resizeWindow('frame',300,300)
                c = 0
                frame_width = int(cam.get(3))
                frame_height = int(cam.get(4))
   
                size = (frame_width, frame_height)
                
                # result = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc(*'XVID'),20.0,size)
                # cv2.resizeWindow('frame',300,300)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_name = filename.split('.')[0]
                result = cv2.VideoWriter('output_' + str(out_name) + '.mp4', fourcc, 20.0, (frame_width, frame_height))
                
                # width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
                # height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
                # size = (width, height)
                # fourcc = cv2.VideoWriter_fourcc(*'XVID')
                # result = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
                frame_num = 0
                # face_arr = []
                maskNet = load_model(r'mask_detector.model')
                fields = ['Frame Number', 'Total non-masked faces', 'Total masked faces', 'Non-masked Face ROIs', 'Masked Face ROIs'] 

                # data rows of csv file 
                rows = []
                face_ids = {}

                while True:
                    final_faces = []
                    face_arr = []
                    addtional_attribute_list = []
                    ret, frame = cam.read()
                    if not ret:
                        logger.warning("ret false")
                        break
                    if frame is None:
                        logger.warning("frame drop")
                        break
                    frame_num += 1
                    # inside frame loop
                    masked = []
                    nonmasked = []
                    frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
                    # vidout=cv2.resize(frame,(300,300)) #create vidout funct. with res=300x300
                    result.write(frame) #write frames of vidout function
                    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if c % detect_interval == 0:
                        img_size = np.asarray(frame.shape)[0:2]
                        mtcnn_starttime = time()
                        faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,
                                                                factor)
                        logger.info("MTCNN detect face cost time : {} s".format(
                            round(time() - mtcnn_starttime, 3)))  # mtcnn detect ,slow
                        face_sums = faces.shape[0]
                        if face_sums > 0:
                            face_list = []
                            for i, item in enumerate(faces):
                                score = round(faces[i, 4], 6)
                                if score > face_score_threshold:
                                    det = np.squeeze(faces[i, 0:4])

                                    # face rectangle
                                    det[0] = np.maximum(det[0] - margin, 0)
                                    det[1] = np.maximum(det[1] - margin, 0)
                                    det[2] = np.minimum(det[2] + margin, img_size[1])
                                    det[3] = np.minimum(det[3] + margin, img_size[0])
                                    face_list.append(item)

                                    # face=frame[startY:endY, startX:endX]
                                    # face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                                    # face=cv2.resize(face,(224,224))
                                    # face=img_to_array(face)
                                    # face=preprocess_input(face)

                                    facex=frame[int(det[1]):int(det[3]), int(det[0]):int(det[2])]
                                    facex=cv2.cvtColor(facex,cv2.COLOR_BGR2RGB)
                                    facex=cv2.resize(facex,(224,224))
                                    facex=img_to_array(facex)
                                    facex=preprocess_input(facex)

                                    # print("facex = ", facex)
                                    face_arr.append(facex)
                                
                                    # face cropped
                                    bb = np.array(det, dtype=np.int32)

                                    # use 5 face landmarks  to judge the face is front or side
                                    squeeze_points = np.squeeze(points[:, i])
                                    tolist = squeeze_points.tolist()
                                    facial_landmarks = []
                                    for j in range(5):
                                        item = [tolist[j], tolist[(j + 5)]]
                                        facial_landmarks.append(item)
                                    if args.face_landmarks:
                                        for (x, y) in facial_landmarks:
                                            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                                    cropped = frame[bb[1]:bb[3], bb[0]:bb[2], :].copy()

                                    dist_rate, high_ratio_variance, width_rate = judge_side_face(
                                        np.array(facial_landmarks))

                                    # face addtional attribute(index 0:face score; index 1:0 represents front face and 1 for side face )
                                    item_list = [cropped, score, dist_rate, high_ratio_variance, width_rate]
                                    addtional_attribute_list.append(item_list)

                            final_faces = np.array(face_list)
                            
                            if len(face_arr)>0:
                                face_arr = np.array(face_arr,dtype='float32')
                                preds = maskNet.predict(face_arr,batch_size=12)
                                temp = []
                                for i, item in enumerate(addtional_attribute_list):
                                    item.append(preds[i][0])
                                    item.append(preds[i][1])
                                    temp.append(item)

                                addtional_attribute_list = temp
                            # print('******************************')
                            # print(final_faces)
                            # print('******************************')

                    trackers = tracker.update(final_faces, img_size, directoryname, addtional_attribute_list, detect_interval)

                    c += 1

                    for d in trackers:
                        if not no_display:
                            d_mask = d
                            d = d.astype(np.int32)
                            # print("d = ", d_mask)
                            mask = d_mask[5]
                            withoutMask = d_mask[6]
                            
                            # Computing the time stamp
                            start_time = 1/fps * frame_num
                            end_time = 1/fps * (frame_num + 1)
                            
                            for item in d_mask:    
                                if item in face_ids.keys():
                                    face_ids[item][1] = end_time
                                else:
                                    face_ids[item] = [start_time, end_time]
                            
                            #determine the class label and color we will use to draw the bounding box and text
                            label='Mask' if mask>withoutMask else 'No Mask'
                            color=(0,255,0) if label=='Mask' else (0,0,255)

                            if mask > withoutMask:
                                masked.append([d[0],d[1],d[2]-d[0],d[3]-d[1]])
                            else:
                                nonmasked.append([d[0],d[1],d[2]-d[0],d[3]-d[1]])
                            
                            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), color, 3)
                            if final_faces != []:
                                cv2.putText(frame, 'ID : %d %s' % (d[4], label), (d[0] - 10, d[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.75,
                                            color, 2)
                                # cv2.putText(frame, 'DETECTOR', (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(1, 1, 1), 2)
                            else:
                                cv2.putText(frame, 'ID : %d' % (d[4]), (d[0] - 10, d[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.75,
                                            colours[d[4] % 32, :] * 255, 2)

                    maskedstr = ';'.join(','.join('%d' %x for x in face) for face in masked)
                    nonmaskedstr = ';'.join(','.join('%d' %x for x in face) for face in nonmasked)
                    rows.append([frame_num, len(nonmasked), len(masked), nonmaskedstr, maskedstr])

                    if not no_display:
                        frame = cv2.resize(frame, (0, 0), fx=show_rate, fy=show_rate)
                        # print('****************')
                        # # result.write(frame)
                        # print(result.write(frame))
                        # print('*******************')
                        vidout=cv2.resize(frame,(frame_width, frame_height)) #create vidout funct. with res=300x300
                        result.write(vidout) #write frames of vidout function
                        
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                result.release()
                # name of csv file 
                filename = str(out_name) + ".csv"
                new = pd.DataFrame.from_dict(face_ids)
                # np.savetxt(r'c:\data\np.txt', df.values, fmt='%d')
                with open(str(out_name) + '.txt', 'w') as f:
                    dfAsString = new.to_string(header=False, index=False)
                    f.write(dfAsString)
                
                with open(str(out_name) + '.txt', 'w') as convert_file:
                    convert_file.write(json.dumps(face_ids))
                
                # writing to csv file 
                with open(filename, 'w') as csvfile: 
                    # creating a csv writer object 
                    csvwriter = csv.writer(csvfile) 
                        
                    # writing the fields 
                    csvwriter.writerow(fields) 
                        
                    # writing the data rows 
                    csvwriter.writerows(rows)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.', default='videos')
    parser.add_argument('--output_path', type=str,
                        help='Path to save face',
                        default='facepics')
    parser.add_argument('--detect_interval',
                        help='how many frames to make a detection',
                        type=int, default=1)
    parser.add_argument('--margin',
                        help='add margin for face',
                        type=int, default=10)
    parser.add_argument('--scale_rate',
                        help='Scale down or enlarge the original video img',
                        type=float, default=0.7)
    parser.add_argument('--show_rate',
                        help='Scale down or enlarge the imgs drawn by opencv',
                        type=float, default=1)
    parser.add_argument('--face_score_threshold',
                        help='The threshold of the extracted faces,range 0<x<=1',
                        type=float, default=0.85)
    parser.add_argument('--face_landmarks',
                        help='Draw five face landmarks on extracted face or not ', action="store_true")
    parser.add_argument('--no_display',
                        help='Display or not', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
