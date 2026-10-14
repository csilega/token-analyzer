from locust import HttpUser, task, constant


class RateLimitProbeUser(HttpUser):
    # No wait between requests -- we want to exceed 10/minute as fast
    # as possible, not simulate realistic human pacing.
    wait_time = constant(0)

    @task
    def hit_analyze(self):
        self.client.post("/analyze", json={"text": "rate limit probe", "model": "gpt-4"})