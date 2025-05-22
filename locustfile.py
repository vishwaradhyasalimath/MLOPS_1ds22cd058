from locust import HttpUser, task, between
import random

class InferenceUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        features = [random.uniform(4.3, 7.9), random.uniform(2.0, 4.4), random.uniform(1.0, 6.9), random.uniform(0.1, 2.5)]
        self.client.post("/predict", json={"features": features})