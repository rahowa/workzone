from flask import Flask
from config import MainConfig
from .extensions import mongo

from .views import main


def create_app(config):
    app = Flask(__name__)
    app.config.from_object(MainConfig)
    mongo.init_app(app)
    app.register_blueprint(main)
    return app
