from typing import Any, Optional
from flask import Response, render_template, Blueprint

from app.drawer import (get_keypoints_drawer, get_workzone_drawer, 
                        get_face_detection_drawer, get_segmentation_drawer,
                        get_object_detection_drawer)
from app.camera import get_video_capture


bp_streams = Blueprint("blueprint_streams", __name__)


@bp_streams.route('/video_feed/<operation>')
def video_feed(operation: str) -> Response:
    """
    Parameters
    ----------
        operation, str
            Define image processing operation.
            Possible variants:
            ze  - workzone estimation
            od  - object detection;
            seg - segmentation;
            fd  - face detection;
            fr  - face recognition;
            kp  - keypoint detection
    Return
    ------
        Response with modified frame from camera
    """

    capture = get_video_capture(0)
    drawer: Optional[Any] = None
    if operation == "ze":
        drawer = get_workzone_drawer()
    elif operation == "fd":
        drawer = get_face_detection_drawer()
    elif operation == "seg":
        drawer = get_segmentation_drawer()
    elif operation == "od":
        drawer = get_object_detection_drawer()
    elif operation == "kp":
        drawer = get_keypoints_drawer()
    return Response(capture.gen_encoded_frame(drawer),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@bp_streams.route('/workzone_stream/<operation>')
def stream(operation: str):
    return render_template('stream.html', operation=operation)
