# Use a base image
FROM python:3.10-slim

# Set the working directory
WORKDIR / E:\VScode projects\football_predictions\model_2

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the application files
COPY . .

# Command to run the application
CMD ["python", "ml.py"]