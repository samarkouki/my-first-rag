from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class Moderator:

    def __init__(self):
        self.client = Groq()