from flask import Flask
from config import MainConfig
from .extensions import mongo

def create_app(config):
    app = Flask(__name__)
    app.config.from_object(config)
    mongo.init_app(app)
    return app 

app = create_app(MainConfig)

from app import views
from app import stream_views