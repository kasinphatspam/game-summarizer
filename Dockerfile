# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# If you have dependencies, copy requirements.txt and install them
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy main.py into the container
COPY main.py /app

# Make port 80 available to the world outside this container (optional)
EXPOSE 80

# Run main.py when the container launches
CMD ["python", "main.py"]
