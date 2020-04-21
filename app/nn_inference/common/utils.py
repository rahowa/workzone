import cv2
from typing import List, Any
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

