# Use a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /model_3

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application files
COPY . .

# Command to run the application
CMD ["python", "club league.py"]