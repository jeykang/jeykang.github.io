import os

# Required to start Superset
SECRET_KEY = os.environ.get("SUPERSET_SECRET_KEY", "this_is_a_very_secret_key")

# Force metadata DB to Postgres when provided via compose
SQLALCHEMY_DATABASE_URI = os.environ.get(
    "SUPERSET__SQLALCHEMY_DATABASE_URI",
    "sqlite:////app/superset_home/superset.db?check_same_thread=false",
)
