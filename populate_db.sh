#!/bin/bash
set -e

# Populate the database with image locations
find /var/lib/images -type f \( -name "*.jpg" -o -name "*.png" \) | while read file; do
  filename=$(basename "$file")
  filepath=$(realpath "$file")
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    INSERT INTO video_frames (filename, filepath) VALUES ('$filename', '$filepath');
EOSQL
done
