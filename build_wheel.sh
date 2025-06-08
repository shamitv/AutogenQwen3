#!/bin/bash
set -e

# Clean previous builds
rm -rf dist build *.egg-info

# Build wheel
python3 -m pip install --upgrade build
python3 -m build --wheel

echo "Wheel file created in the dist/ directory."

