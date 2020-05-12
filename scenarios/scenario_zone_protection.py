import cv2
import asyncio
import aiohttp
import numpy as np
from typing import List, Tuple, Dict

from app.base_types import Image
from app.nn_inference.common.utils import draw_keypoints
from robot_work_zone_estimation.src.workzone import is_inside, Point


async def get_workzone(session: aiohttp.ClientSession,
                       url: str,
                       data: str,
                       headers: Dict[str, str]) -> List[Tuple[int, int]]:
    async with session.post(url, data=data, headers=headers) as resp:
        result = await resp.json()
    return result["workzone"]


async def get_keypoints(session: aiohttp.ClientSession,
                        url: str,
                        data: str,
                        headers: Dict[str, str]) -> List[List[Tuple[float, float, float]]]:
    async with session.post(url, data=data, headers=headers) as resp:
        result = await resp.json()
    return result["keypoints"]["keypoints"]


def draw_workzone(frame, zone_polygon) -> Image:
    zone_polygon = np.array(zone_polygon).reshape(-1, 1, 2)
    zone_polygon = np.clip(zone_polygon, 0, np.inf)
    frame = cv2.polylines(frame, [np.int32(zone_polygon)], True, (255, 0, 0), 2, cv2.LINE_AA)
    return frame


def alarm(polygon: List[Tuple[int, int]],
               keypoints: List[List[Tuple[float, float, float]]]) -> None:
    polygon = tuple(Point(point[0], point[1]) for point in polygon)
    for persons_kp in keypoints:
        for point in persons_kp:
            if point[2]:
                if is_inside(polygon, Point(point[0], point[1])):
                    print("ALARM! PERSON IS IN DANGER ZONE!")


async def main_zone_protection(address: str) -> None:
    wz_route = "workzone"
    kp_route = "object/keypoint"
    wz_url = f"{address}/{wz_route}"
    kp_url = f"{address}/{kp_route}"

    content_type = 'image/jpeg'
    headers = {'content-type': content_type}

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    async with aiohttp.ClientSession() as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, image_to_send = cv2.imencode('.jpg', frame)
            polygons_and_kp = await asyncio.gather(get_keypoints(session, kp_url, image_to_send.tostring(), headers),
                                                   get_workzone(session, wz_url, image_to_send.tostring(), headers))
            keypoints, zone_polygon = polygons_and_kp[0], polygons_and_kp[1]
            alarm(zone_polygon, keypoints)
            frame = draw_keypoints(frame, keypoints)
            frame = draw_workzone(frame, zone_polygon)
            cv2.imshow("test case", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()