#!/bin/bash

# List all files in the repository, including those in other branches
files=$(git log --all --name-only --pretty=format: | sort -u)

for file in $files; do
    # Get the latest modification date for the file across all branches
    latest_commit=$(git log --all -1 --format="%ci %h %d %s" -- "$file")
    if [ -n "$latest_commit" ]; then
        echo "$latest_commit $file"
    fi
done
