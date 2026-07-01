FROM python:3.12-slim-trixie
WORKDIR /app
# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-dev libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better layer caching)
COPY torch-requirements.txt requirements.txt ./
RUN pip install -r torch-requirements.txt
RUN pip install -r requirements.txt

# Copy the rest of the application code last
COPY . .

EXPOSE 8080
CMD ["python", "app.py"]
