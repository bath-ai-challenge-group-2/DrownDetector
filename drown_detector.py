import cv2
import time
import torch
import multiprocessing as mp

from data_service import *
from data_service.sources import *
from data_service.services import *
from data_service.outputs import *
from river_segmentation import *

from flask import Flask, render_template, Response


if __name__=="__main__":
    FPS = 30

    torch.multiprocessing.set_start_method('spawn')
    mp.set_start_method('spawn', force=True)

    app = Flask('MainApp', template_folder='./web_template')
    camera = WebServerOutput(30, 30, (1920, 1080))

    # segmentation_map = DummySegmentation(x=400)
    segmentation_map = MaskSegmentation('./input_files/5053_mask_image_modified.png')

    pipeline = DataPipeline()

    # Create Video Ingestion Service
    video_ingestion = VideoSource('./input_files/IMG_5053.MOV', buffer_size=30)
    # ip_camera = IPCamera(address='tcp://localhost:40001', buffer_size=FPS, img_dim=(480, 720, 3))
    yolo_detection = YoloV5Detection()
    person_extraction = PeopleExtraction()
    # pose_estimation = AlphaPoseEstimation()
    person_tracker = PeopleTracker(distance_threshold=200)
    drown_detector = DrownDetection(segmentation_map)

    pipeline.add_service(video_ingestion)
    # pipeline.add_service(ip_camera)
    pipeline.add_service(yolo_detection)
    pipeline.add_service(person_extraction)
    # pipeline.add_service(pose_estimation)
    pipeline.add_service(person_tracker)
    pipeline.add_service(drown_detector)
    pipeline.add_service(camera)

    pipeline.start()


    @app.route('/')
    def index():
        camera.release_video()
        #return render_template('index.html')
        return {}


    def gen(camera):
        while True:
            # s_time = time.time()
            frame = camera.get_frame()
            _, frame = cv2.imencode('.jpg', frame)
            # print('Frame Time: {}s'.format(time.time() - s_time))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')


    @app.route('/video_feed')
    def video_feed():
        return Response(gen(camera),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=8888)


    # while True:
    #     if len(yolo_detection) < 1:
    #         continue
    #
    #     print("Recieved new Frames")
    #
    #     frame_buffer = yolo_detection.get_next()
    #
    #     cv2.destroyAllWindows()
    #

    # while True:
    #     if len(video_ingestion) < 1:
    #         continue
    #
    #     print("Recieved new Frames")
    #
    #     frame_buffer = video_ingestion.get_next()
    #
    #     for i in range(frame_buffer.buffer_length):
    #         frame = frame_buffer.get()[i]
    #
    #         cv2.imshow('image', frame)
    #         cv2.waitKey(0)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     cv2.destroyAllWindows()
