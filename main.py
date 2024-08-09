import cv2
import base64
import requests
import numpy as np
import utils as utl
from datetime import datetime
from multiprocessing import Process, Lock, Queue

depth_detect_url = "http://0.0.0.0:8001/draw_depth/"
face_detect_url = "http://0.0.0.0:8002/draw_face/"
object_detect_url = "http://0.0.0.0:8003/draw_object/"

def call_img_func(url: str, json: dict, timeout: float=1.0):

    try:
        if not (url is None or json is None):

            response = requests.post(url=url, headers={'Content-Type': 'application/json'}, json=json, timeout=timeout)
            if response.status_code == 200:  

                response = response.json()
                return response
            
    except Exception as e:
        print(e, flush=True)

    return None

def show_window(bgr_image_queue):

    showing = False
    while not bgr_image_queue is None:
        try:
            image = bgr_image_queue.get(False)
            if not(image['title'] is None or image['frame'] is None):
                cv2.imshow(image['title'], image['frame'])
                showing = True

        except Exception as e:
            print(e, flush=True)

        if showing is True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if showing is True:
        cv2.destroyAllWindows()

    if not bgr_image_queue is None:
        bgr_image_queue.close()
    
def red_shift(image: np.ndarray, shift_factor: float=0.5):

    try:
        output_image = image.copy()
        output_image[:,:,0] = output_image[:,:,0] * (1 - shift_factor)
        output_image[:,:,1] = output_image[:,:,1] * (1 - shift_factor)
        return output_image

    except Exception as e:
        print(e, flush=True)

    return image    

def apply_noise(image: np.ndarray, noise_factor: float=0.25):

    try:
        output_image = image.copy()
        noise_factor = np.clip(noise_factor, 0, 1)
        noise = np.random.randint(0, 256, (output_image.shape[0], output_image.shape[1]), dtype = np.uint8)
        noise = np.repeat(noise[:, :, np.newaxis], 3, axis=2)
        output_image = cv2.addWeighted(output_image, 1 - noise_factor, noise, noise_factor, 0)
        return output_image

    except Exception as e:
        print(e, flush=True)  

    return image

def apply_horizontal_grid(image: np.ndarray, line_spacing: int=6, line_color: list=(0, 0, 0), line_thickness: int=1):

    try:
        output_image = image.copy()
        height = image.shape[0]
        for y in range(0, height, line_spacing):

            cv2.line(output_image, (0, y), (image.shape[1], y), line_color, line_thickness)

        return output_image

    except Exception as e:
        print(e, flush=True)  

    return image

def detect_face(jpg_str: str, shape, gpu, bgr_image_queue):

    try:
        if gpu.acquire(block=False) and not bgr_image_queue is None:

            results = call_img_func(url=face_detect_url, json={'rgb_image': jpg_str})
            gpu.release()
            if not results is None:

                image = np.zeros(shape=shape, dtype=np.uint8)
                image = utl.draw_landmarks(image, results['face_landmarks'])
                bgr_image_queue.put({'title': 'Face mapping', 'frame': image}, timeout=0.1)
                        
    except Exception as e:
        print(e, flush=True)

def detect_depth(jpg_str: str, gpu, bgr_image_queue):

    try:
        if gpu.acquire(block=False) and not bgr_image_queue is None:

            image = call_img_func(url=depth_detect_url, json={'rgb_image': jpg_str})
            gpu.release()
            if not image is None:

                image = base64.b64decode(image['result_frame'])
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                bgr_image_queue.put({'title': 'Depth mapping', 'frame': image}, timeout=0.05)
            
    except Exception as e:
        print(e, flush=True)

def detect_object(jpg_str: str, gpu, bgr_image_queue):

    try:
        if gpu.acquire(block=False) and not bgr_image_queue is None:

            results = call_img_func(url=object_detect_url, json={'rgb_image': jpg_str})
            gpu.release()
            if not results is None:

                results = results['object_list']
                for idx, object in enumerate(results):

                    mask = np.zeros(shape=(480, 640), dtype=np.uint8)
                    for (x, y) in object[3]:

                        mask[int(y)][int(x)] = 1

                    results[idx][3] = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)

                tape = results[0][3]
                legend = [results[0][1] + ' (track_id: {id})'.format(id=results[0][0])]
                if len(results) > 1:
                    for object in results[1:]:

                        tape = np.hstack(tup=[tape, object[3]])
                        legend = legend.append(object[1] + '(track_id: {id})'.format(id=object[0]))

                if not bgr_image_queue is None:

                    bgr_image_queue.put({'title': 'Object mapping', 'frame': tape, 'legend': legend}, timeout=0.05)
            
    except Exception as e:
        print(e, flush=True)

def show_main_vision(main_image_queue, window_shape: tuple=(1280, 960)):

    try:
        if not main_image_queue is None:

            bkg_frame = None
            face_frame = None
            depth_frame = None
            object_frame = None

            frame_count = 0
            interval = 120
            red_flag = True
            text_color = (0, 255, 255)
            red_text_color = (255, 255, 255)
            noise_factor = 0.0

            labels = [{'text': 'T-800 VISION', 'org': (30, 30), 'fontFace': cv2.FONT_HERSHEY_PLAIN, 'fontScale': 2.5, 'thickness': 2, 'lineType': cv2.LINE_AA},
                      {'text': '_FACE MAPPING', 'org': (850, 60), 'fontFace': cv2.FONT_HERSHEY_PLAIN, 'fontScale': 2.5, 'thickness': 2, 'lineType': cv2.LINE_AA},
                      {'text': '_INVENTORY', 'org': (30, 650), 'fontFace': cv2.FONT_HERSHEY_PLAIN, 'fontScale': 2.5, 'thickness': 2, 'lineType': cv2.LINE_AA},
                      {'text': '_DEPTH MAPPING', 'org': (850, 650), 'fontFace': cv2.FONT_HERSHEY_PLAIN, 'fontScale': 2.5, 'thickness': 2, 'lineType': cv2.LINE_AA}]

            window_image_queue = Queue(1)
            p1 = Process(target=show_window, args=(window_image_queue,))
            p1.start()
            while p1.is_alive():
                try:
                    image = main_image_queue.get(False)
                    if not image['frame'] is None:

                        now = datetime.now()
                        frame_count = frame_count + 1
                        if frame_count % interval == 0:
                            if red_flag is True:
                                red_flag = False
                                interval = 30
                            else:
                                red_flag = True
                                interval = 150

                        if image['title'] == 'Main':

                            bkg_frame = apply_noise(image['frame'], noise_factor)
                            bkg_frame = cv2.resize(bkg_frame, window_shape, interpolation = cv2.INTER_LINEAR)

                            red_bkg_frame = red_shift(image=image['frame'], shift_factor=1.0)
                            red_bkg_frame = apply_horizontal_grid(red_bkg_frame, line_spacing=5, line_thickness=1)
                            red_bkg_frame = cv2.resize(red_bkg_frame, window_shape, interpolation = cv2.INTER_LINEAR)
                        
                        elif image['title'] == 'Object mapping':

                            object_height = int(window_shape[1] * 0.28)
                            object_width = int(image['frame'].shape[1] * (object_height / image['frame'].shape[0]))
                            object_frame = cv2.resize(image['frame'], (object_width, object_height), interpolation=cv2.INTER_AREA)

                            if object_frame.shape[1] > window_shape[0]:
                                object_frame = image['frame'][:, :window_shape[0], :]

                            object_width = object_frame.shape[1]
                            object_legend = image['legend']
                            
                        elif image['title'] == 'Face mapping':

                            face_frame = image['frame']
                            face_height = face_frame.shape[0]
                            face_width = face_frame.shape[1]
                                         
                            face_height = int(window_shape[1] * 0.25)
                            face_width = int(image['frame'].shape[1] * (face_height / image['frame'].shape[0]))
                            face_frame = cv2.resize(image['frame'], (face_width, face_height), interpolation=cv2.INTER_AREA)
                            red_face_frame = cv2.applyColorMap(face_frame, cv2.COLORMAP_HOT)

                        elif image['title'] == 'Depth mapping':

                            depth_height = int(window_shape[1] * 0.33)
                            depth_width = int(image['frame'].shape[1] * (depth_height / image['frame'].shape[0]))
                            depth_frame = cv2.resize(image['frame'], (depth_width, depth_height), interpolation=cv2.INTER_AREA)
                            depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
                            red_depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_HOT)
                            
                    if not bkg_frame is None: 
                        if red_flag is True:
                            noise_factor = 0.0

                            if not object_frame is None:
                                red_bkg_frame[window_shape[1] - object_height - 30:window_shape[1] - 30, 30:object_width + 30][object_frame == 1] = red_text_color

                                x_pos = 50
                                y_pos = window_shape[1] - object_height - 10
                                delta_x = int(window_shape[0] / len(object_legend))
                                for legend in object_legend:

                                    red_bkg_frame = cv2.putText(img=red_bkg_frame, text=legend, org=(x_pos, y_pos), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=red_text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                                    x_pos = x_pos + delta_x

                            if not depth_frame is None:
                                red_bkg_frame[window_shape[1] - depth_height - 30:window_shape[1] - 30, window_shape[0] - depth_width - 30:window_shape[0] - 30] = red_depth_frame

                            if not face_frame is None:
                                red_bkg_frame[60: face_height + 60, window_shape[0] - face_width - 60:window_shape[0] - 60][red_face_frame > 0] = 255

                            red_bkg_frame = cv2.putText(img=red_bkg_frame, text=str(frame_count), org=(30, 90), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=red_text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                            red_bkg_frame = cv2.putText(img=red_bkg_frame, text=now.strftime("%c"), org=(30, 60), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=red_text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                            for label in labels:
                                red_bkg_frame = cv2.putText(img=red_bkg_frame, text=label['text'], org=label['org'], fontFace=label['fontFace'], fontScale=label['fontScale'], color=red_text_color, thickness=label['thickness'], lineType=label['lineType'])
                                                            
                            window_image_queue.put({'title':'Main', 'frame':red_bkg_frame}, timeout=0.05)

                        else:
                            noise_factor = noise_factor + 1/interval

                            if not object_frame is None:
                                bkg_frame[window_shape[1] - object_height - 30:window_shape[1] - 30, 30:object_width + 30][object_frame == 1] = text_color

                                x_pos = 50
                                y_pos = window_shape[1] - object_height - 10
                                delta_x = int(window_shape[0] / len(object_legend))
                                for legend in object_legend:

                                    bkg_frame = cv2.putText(img=bkg_frame, text=legend, org=(x_pos, y_pos), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                                    x_pos = x_pos + delta_x

                            if not depth_frame is None:
                                bkg_frame[window_shape[1] - depth_height - 30:window_shape[1] - 30, window_shape[0] - depth_width - 30:window_shape[0] - 30] = depth_frame

                            if not face_frame is None:
                                bkg_frame[60: face_height + 60, window_shape[0] - face_width - 60:window_shape[0] - 60, 0][red_face_frame[:, :, 0] > 0] = 0
                                bkg_frame[60: face_height + 60, window_shape[0] - face_width - 60:window_shape[0] - 60, 1:2][red_face_frame[:, :, 1:2] > 0] = 255

                            bkg_frame = cv2.putText(img=bkg_frame, text=str(frame_count), org=(30, 90), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                            bkg_frame = cv2.putText(img=bkg_frame, text=now.strftime("%c"), org=(30, 60), fontFace=labels[0]['fontFace'], fontScale=labels[0]['fontScale'], color=text_color, thickness=labels[0]['thickness'], lineType=labels[0]['lineType'])
                            for label in labels:
                                bkg_frame = cv2.putText(img=bkg_frame, text=label['text'], org=label['org'], fontFace=label['fontFace'], fontScale=label['fontScale'], color=text_color, thickness=label['thickness'], lineType=label['lineType'])

                            window_image_queue.put({'title':'Main', 'frame':bkg_frame}, timeout=0.05)

                except Exception as e:
                    print(e, flush=True)

    except Exception as e:
        print(e, flush=True)
                    
def main():
    
    cam_id = 0
    while cam_id < 4:
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            break
        cam_id = cam_id + 1

    if cap.isOpened():
        bgr_image_queue = Queue(4)
        p2 = Process(target=show_main_vision, args=(bgr_image_queue,))
        p2.start()        

        gpu = Lock()
        while p2.is_alive() and not bgr_image_queue is None:
            try:
                ret, cap_frame = cap.read()
                if ret:
            
                    ret, jpg_frame = cv2.imencode(ext='.jpg', img=cap_frame)
                    if ret:

                        shape = cap_frame.shape
                        jpg_str = base64.b64encode(jpg_frame).decode('utf-8')
                        p3 = Process(target=detect_object, args=(jpg_str, gpu, bgr_image_queue,))
                        p4 = Process(target=detect_depth, args=(jpg_str, gpu, bgr_image_queue,))
                        p5 = Process(target=detect_face, args=(jpg_str, shape, gpu, bgr_image_queue,))
                        p3.start()
                        p4.start()
                        p5.start()

                    bgr_image_queue.put({'title': 'Main', 'frame': cap_frame}, timeout=0.001)

            except Exception as e:
                print(e, flush=True)

        p5.terminate()
        p4.terminate()
        p3.terminate()
        p2.terminate()

    cap.release()


if __name__ == "__main__":
    main()