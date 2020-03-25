from flask import Response, render_template, Blueprint

from app.drawer import get_drawer
from app.camera import get_video_capture


bp_streams = Blueprint("blueprint_streams", __name__)

@bp_streams.route('/video_feed')
def video_feed():
    capture = get_video_capture(0)
    drawer = get_drawer("/Users/valentinevseenko/Training/workzone/robot_work_zone_estimation/YOUR_CUSTOM_CONFIG.json")
    return Response(capture.gen_encoded_frame(drawer),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@bp_streams.route('/workzone_stream')
def stream():
    return render_template('stream.html')