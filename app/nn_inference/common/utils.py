from app.nn_inference import keypoints
import cv2
import numpy as np
from typing import List, Any, Tuple
from app.base_types import Image, Boxes


def draw_bboxes(image: Image, bboxes: Boxes) -> Image:
    """ Draw bbox in format (xmin, ymin, xmax, ymax)
        over the image

        Parameters
        ----------
            image: Image, np.ndarray
                Image with faces or object
            bboxes: Bboxes
                BBoxes of founded faces or objects on image
        Returns
        -------
            image: image with drawed bboxes
    """
    color = (255, 255//2, 255//3)
    for bbox in bboxes:
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[2], bbox[3])
        image = cv2.rectangle(image, pt1, pt2, color=color, thickness=2)
    return image


def chunks(lst: List[Any], n: int):
    """Yield successive n-sized chunks from lst
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def decode_segmap(image, nc=21):
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    print(image.shape, r.shape, g.shape, b.shape)
    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def draw_keypoints(image: Image, keypoints: List[List[Tuple[float, float, float]]]) -> Image:
    color = (255//2, 255//2, 0)
    for person in keypoints:
        for kp in person:
            if kp[2]:
                center = (int(kp[0]), int(kp[1]))
                image = cv2.circle(image, center, 4, color, 2)
    return image