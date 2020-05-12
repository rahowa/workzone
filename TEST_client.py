import cv2
import numpy as np
import requests

from app.nn_inference.common.utils import draw_bboxes


"""
    Original implementation at 
    https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
"""


operation = "workzone"
addr = 'http://0.0.0.0:1000'
test_url = addr + f'/{operation}'
path_to_test_image = "./test_image.jpg"

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True: 
        ret, frame = cap.read()

        if not ret:
            break

        _, image_to_send = cv2.imencode('.jpg', frame)
        response = requests.post(test_url, data=image_to_send.tostring(),
                                 headers=headers)
        if operation == "detect":
            boxes = response.json()["faces_loc"]
            if len(boxes) > 0:
                print(boxes)
                frame = draw_bboxes(frame, boxes)

        elif operation == "recognize":
            ids = response.json()["faces_id"]
            print(ids)
        else:
            zone_polygon = response.json()["workzone"]
            zone_polygon = np.array(zone_polygon).reshape(-1, 1, 2)
            zone_polygon = np.clip(zone_polygon, 0, np.inf)
            frame = cv2.polylines(frame, [np.int32(zone_polygon)], True, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("test case", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
