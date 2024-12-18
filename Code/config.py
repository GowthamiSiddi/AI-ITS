from dotenv import load_dotenv
import os

# load configuration fot environment variables
def load_config():
    return load_dotenv()

def get_google_api_key():
    #GOOGLE_API_KEY="AIzaSyDuwNT8YYue3otvFVTI9g4PiOrJgPznr6Q"
    return os.getenv("GOOGLE_API_KEY")