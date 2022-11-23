# import libraries
import os
from configparser import ConfigParser

# initialise env variable
env = os.getenv("ENV", "./.config")
if env == "./.config":
    config = ConfigParser()
    config.read("./.config")
    config = config["AUTH0"]
    # get var for token validation
    SECRET_TOKEN = config["SECRET_TOKEN"]


# func for token validation
def is_actual_credentials(token):
    return token == SECRET_TOKEN
