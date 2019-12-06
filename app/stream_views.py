from flask import Response, render_template, g

from app import app
from app.camera import (CamReader, Image,
                        wrap_image, encode_image)
from app.drawer import DrawZone


def get_video_capture(cam_id: int) -> CamReader:
    caputure = getattr(g, "_capture", None)
    if caputure is None:
        caputure = g._capture = CamReader(cam_id)
    return caputure


def get_drawer(config_path: str) -> DrawZone:
    drawer = getattr(g, "_zone_drawer", None)
    if drawer is None:
        drawer = g._zone_drawer = DrawZone(config_path)
    return drawer


@app.route('/video_feed')
def video_feed():
    capture = get_video_capture(0)
    drawer = get_drawer("/Users/valentinevseenko/Training/workzone/robot_work_zone_estimation/YOUR_CUSTOM_CONFIG.json")
    return Response(capture.gen_encoded_frame(drawer),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/workzone_stream')
def stream():
    return render_template('stream.html')