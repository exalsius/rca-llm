FROM locustio/locust

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["locust", "-f", "main.py", "--config", "locust.conf"]