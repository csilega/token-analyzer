import random
import uuid

from locust import HttpUser, task, between

MODELS = ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo", "bert-base-uncased", "gpt2"]

CACHE_FRIENDLY_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "To be or not to be, that is the question.",
    "def hello_world():\n    print('hello, world')",
]

NEAR_CAP_TEXT = "a" * 500_000


class TokenAnalyzerUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(5)
    def analyze_cache_friendly(self):
        self.client.post("/analyze", json={"text": random.choice(CACHE_FRIENDLY_TEXTS), "model": random.choice(MODELS)})

    @task(3)
    def analyze_unique(self):
        self.client.post("/analyze", json={"text": f"Unique request payload {uuid.uuid4()}", "model": random.choice(MODELS)})

    @task(1)
    def analyze_near_cap(self):
        self.client.post("/analyze", json={"text": NEAR_CAP_TEXT, "model": random.choice(MODELS)})

    @task(2)
    def compare(self):
        self.client.post("/compare", json={"text": random.choice(CACHE_FRIENDLY_TEXTS), "models": MODELS})