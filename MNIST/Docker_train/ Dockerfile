# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed package specified in requirements.txt
RUN pip install --trusted-host pypi.python.org numpy tensorflow keras

COPY train.y ./train.py

# Run train.py when the container launches
ENTRYPOINT ["python", "train.py"]
