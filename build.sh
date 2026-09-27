#!/bin/bash
# Build script for Vercel

# Install Python dependencies
pip install -r requirements.txt

# Ensure static files are in place
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo "Build completed successfully"
