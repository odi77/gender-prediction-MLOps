# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY . /app

# Install Poetry and dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Expose the port that your Flask app will run on
EXPOSE 5000

# Define environment variable
ENV FLASK_APP main.py

# Run the Flask app when the container launches
CMD ["bash", "-c", "flask --app api run --host 0.0.0.0 --port $FLASK_PORT"]

