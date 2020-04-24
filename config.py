import os
from dotenv import load_dotenv


dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(verbose=True)


class MainConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY') or "BNTU_THE_BEST_1488"
    MONGO_URI = os.environ.get("MONGO_URI") or "localhost"
    MONGO_DBNAME = os.environ.get("MONGO_DBNAME") or 'workers'