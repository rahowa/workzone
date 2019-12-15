import os
from dotenv import load_dotenv

load_dotenv(verbose=True)
basedir = os.path.abspath(os.path.dirname(__file__))


class MainConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or "BNTU_THE_BEST_1488"
    MONGO_URI = os.environ.get("MONGO_URI") or "localhost"
