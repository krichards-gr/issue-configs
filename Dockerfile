# Use a python base image
FROM python:3.11-slim

# Enable unbuffered logging
ENV PYTHONUNBUFFERED=1

# Install system dependencies (build-essential for spacy/transformers if needed)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the application and models
# (Models are now downloaded via cloudbuild.yaml before this step)
COPY . /app
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Set the command to run the production script
CMD ["python", "analysis.py"]
