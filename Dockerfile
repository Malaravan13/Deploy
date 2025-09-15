# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY gbr_jaundice_model.joblib .
COPY gbr_feature_names.joblib .

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
