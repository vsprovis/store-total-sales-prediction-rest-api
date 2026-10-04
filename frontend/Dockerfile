# Use a minimal base image with Python 3.11 installed
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the Streamlit app on port 7860 (Hugging Face Spaces default) and make it accessible externally
# XSRF protection is disabled so that file uploads work for batch predictions
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
