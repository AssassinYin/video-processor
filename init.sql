-- Create a table to store image locations and metadata
CREATE TABLE video_frames (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    filepath VARCHAR(255) NOT NULL,
    frame_number INT,
    "timestamp" FLOAT,
    ocr_text TEXT,
    scene_id INT
);
