import cv2
import requests
import base64

from app.nn_inference.common.utils import decode_segmap
from app.image_decoding_utils import deocode_image


def main_person_segmentation(address: str) -> None:
    route = "object/segmentation"
    test_url = f"{address}/{route}"
    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, image_to_send = cv2.imencode('.jpg', frame)
        response = requests.post(test_url,
                                 data=image_to_send.tostring(),
                                 headers=headers)
        detections = response.json()["mask"]
        mask = base64.decodebytes(detections["mask"].encode("utf-8"))
        mask = deocode_image(mask)
        cv2.imshow("test case", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
