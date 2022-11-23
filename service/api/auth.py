# import libraries
import os
from configparser import ConfigParser

# initialise env variable
SECRET_TOKEN = os.getenv("SECRET_TOKEN")
if SECRET_TOKEN is None:
    config = ConfigParser()
    config.read("./.config")
    params = config["AUTH0"]
    # get var for token validation
    SECRET_TOKEN = params["SECRET_TOKEN"]


# func for token validation
def is_actual_credentials(token) -> bool:
    return token == SECRET_TOKEN
