
import cv2
import json
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
    img = cv2.imread('./assets/testface.jpg')
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', img)
    # send http request with image and receive response
    response = requests.post(test_url, data=img_encoded.tostring(),
                             headers=headers)
    # decode response
    print(json.loads(response.text))
