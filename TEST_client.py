import cv2
import json
import time
import requests

from app.nn_inference.common.utils import draw_bboxes


"""
    Original implementation at 
    https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
"""


operation = "recognize"
addr = 'http://0.0.0.0:1000'
test_url = addr + f'/face/{operation}'
path_to_test_image = "./test_image.jpg"

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        # time.sleep(1)
        _, frame = cap.read()
        _, image_to_send = cv2.imencode('.jpg', frame)
        response = requests.post(test_url, data=image_to_send.tostring(),
                                 headers=headers)
        if operation == "detect":
            boxes = response.json()["faces_loc"]
            if len(boxes) > 0:
                print(boxes)
                frame = draw_bboxes(frame, boxes)

        else:
            ids = response.json()["faces_id"]
            print(ids)

        cv2.imshow("test case", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        print(response.text)

    cap.release()
    cv2.destroyAllWindows()
