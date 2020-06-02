import cv2
import requests
import numpy as np


def main_zone_estimation(address: str) -> None:
    route = "workzone"
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
        zone_polygon = response.json()["workzone"]
        zone_polygon = np.array(zone_polygon).reshape(-1, 1, 2)
        zone_polygon = np.clip(zone_polygon, 0, np.inf)
        frame = cv2.polylines(frame, [np.int32(zone_polygon)], True, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("test case", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
