# 1. Start from an official Python base image
FROM python:3.10-slim

# 2. Install system dependencies, including curl to download the gcloud SDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Download and install the Google Cloud SDK
RUN curl -sSL https://sdk.cloud.google.com | bash -s -- --disable-prompts
ENV PATH /root/google-cloud-sdk/bin:$PATH

# 4. Set the working directory inside the container
WORKDIR /app

# 5. Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code into the container
COPY main.py .

# 7. Expose the port the app runs on
EXPOSE 8080

# 8. Define the command to run your app
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}"]