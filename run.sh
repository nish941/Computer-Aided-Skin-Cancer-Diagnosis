#!/bin/bash

echo "Starting Skin Cancer Diagnosis Web App..."
echo "----------------------------------------"

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/resnet50_model.h5" ]; then
    echo "ERROR: Model file not found!"
    echo "Please place resnet50_model.h5 in models/ directory"
    exit 1
fi

# Run the Flask app
echo "Launching application..."
echo "Open http://localhost:5000 in your browser"
python app/app.py
