#!/bin/bash
echo "Running lint checks..."
flake8 src api pipelines
