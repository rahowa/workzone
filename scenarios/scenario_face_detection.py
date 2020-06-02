import cv2
import requests

from app.nn_inference.common.utils import draw_bboxes


def main_face_detection(address: str) -> None:
    route = "face/detect"
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
        boxes = response.json()["faces_loc"]
        if len(boxes) > 0:
            frame = draw_bboxes(frame, boxes)

        cv2.imshow("test case", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
