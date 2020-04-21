from flask import Flask
from config import MainConfig
from .extensions import mongo

from .views import bp_main
from .errors import bp_errors
from .stream_views import bp_streams


def create_app(config=MainConfig):
    app = Flask(__name__)
    app.config.from_object(MainConfig)
    mongo.init_app(app)
    app.register_blueprint(bp_main)
    app.register_blueprint(bp_errors)
    app.register_blueprint(bp_streams)
    return app
