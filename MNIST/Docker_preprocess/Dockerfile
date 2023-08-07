# Use an official Python runtime as a parent image
FROM python:3.7-slim

# Set the working directory in the container to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org numpy tensorflow keras

COPY preprocess_data.py ./preprocess_data.py

# Run preprocess.py when the container launches
ENTRYPOINT [ "python", "preprocess_data.py" ]
