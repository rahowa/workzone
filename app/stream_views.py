from flask import Response, render_template

from app import app
from app.camera import CamReader




@app.route('/video_feed')
def video_feed():
    return Response(CamReader(0).gen_encoded_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream')
def stream():
    return render_template('stream.html')

