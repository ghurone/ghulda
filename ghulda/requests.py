from random import choice

import requests
from requests.adapters import HTTPAdapter, Retry


class Request:
    def __init__(self, useragents_path, n_retry: int = 20, timeout: int = 15, headers: dict = None):
        self.retries = Retry(total=n_retry, backoff_factor=1, status_forcelist=[403, 429, 500, 502, 503, 504])
        self.adapter = HTTPAdapter(max_retries=self.retries)
        self.http = requests.Session()
        self.http.mount("https://", self.adapter)

        self.timeout = timeout

        if headers is None:
            self.headers = {}
        else:
            self.headers = headers

        with open(useragents_path, 'r') as file:
            self.user_agents = [line.split('|')[-1].strip() for line in file if line.strip()]

    def get(self, url: str):
        self.headers["User-Agent"] = choice(self.user_agents)

        return self.http.get(url, timeout=self.timeout, headers=self.headers)

    def post(self, url: str, data=None):
        self.headers["User-Agent"] = choice(self.user_agents)

        return self.http.get(url, timeout=self.timeout, headers=self.headers, data=data)
