#!/bin/bash
set -e

# Clean previous builds
rm -rf dist build *.egg-info

# Build source and binary wheel
python3 -m build --sdist --wheel

echo "Source and binary wheel files created in the dist/ directory."
