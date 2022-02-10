FROM jjanzic/docker-python3-opencv:latest

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "src/app.py"]
