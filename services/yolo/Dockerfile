FROM python:3.12-slim-trixie

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r torch-requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python", "app.py"]
