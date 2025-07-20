# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's source code from your host to your container
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run your app
CMD ["streamlit", "run", "home.py"]
