
import cv2
import json
import time
import requests

"""
    Original implementation at 
    https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
"""

addr = 'http://localhost:5000'
test_url = addr + '/face'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

if __name__ == "__main__":
    # img = cv2.imread('./assets/testface.jpg')
    # _, img_encoded = cv2.imencode('.jpg', img)
    # while True:
    #     print("Send test image")
    #     time.sleep(0.5)
    #     response = requests.post(test_url, data=img_encoded.tostring(),
    #                              headers=headers)
    #     print(json.loads(response.text))

    cap = cv2.VideoCapture(0)
    while True:
        time.sleep(1)
        _, frame = cap.read()
        _, image_to_send = cv2.imencode('.jpg', frame)
        response = requests.post(test_url, data=image_to_send.tostring(),
                                 headers=headers)
        print(json.loads(response.text))