from groq import Groq
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()


class Moderator:

    
    def __init__(self):
        self.client = Groq()

        prompt_path = Path("prompts/moderation_prompt.txt")

        with open(prompt_path, encoding="utf-8") as f:
            self.system_prompt = f.read()