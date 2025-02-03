# Use the official Python image as a base
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT 8000
# Expose the port FastAPI will run on
EXPOSE 8000

# fastapi run main.py --host 0.0.0.0 --port 8000

CMD ["fastapi", "run", "main:app", "--host", "0.0.0.0", "--port", "8000"]