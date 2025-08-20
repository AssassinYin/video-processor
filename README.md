# Video Processing Tool

A powerful Python-based video processing tool that extracts frames from videos with intelligent scene detection, OCR text extraction, and advanced deduplication capabilities.

## Features

- **Scene Detection**: Automatically detects scene changes and extracts keyframes
- **Frame Skipping**: Intelligently skips redundant frames while maintaining content coverage
- **OCR Text Extraction**: Supports both Tesseract and EasyOCR engines
- **Multi-language Support**: Process videos in multiple languages (English, Chinese, etc.)
- **Advanced Deduplication**: Uses SSIM and text similarity for smart frame filtering
- **Multiprocessing**: Parallel processing for improved performance
- **Comprehensive Metadata**: Detailed JSON output with scene information and timestamps

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Basic Frame Extraction
```bash
python video_to_frames.py
```

### Scene Detection Mode (Recommended)
```bash
python video_to_frames.py --scene-detection
```

### Advanced Scene Detection with Custom Settings
```bash
python video_to_frames.py --scene-detection --keyframes-per-scene 5 --scene-threshold 25.0
```

### Chinese Language OCR
```bash
python video_to_frames.py --scene-detection --lang chi_sim
```

## Command Line Options

### Basic Options
- `input_dir`: Directory containing video files (default: `./input`)
- `--output-dir`: Output directory (default: `./output`)
- `--interval`: Extract every nth frame (default: 1)
- `--extensions`: Video file extensions (default: .mp4 .avi .mov)

### Scene Detection Options
- `--scene-detection`: Enable intelligent scene detection
- `--keyframes-per-scene`: Number of keyframes per scene (default: 3)
- `--scene-threshold`: Scene detection sensitivity (default: 30.0)
- `--min-scene-len`: Minimum scene length in frames (default: 15)

### OCR Options
- `--lang`: Language code (default: eng, use chi_sim for Chinese)
- `--text-region`: Text extraction region (all, top, bottom)
- `--ocr-engine`: OCR engine (tesseract, easyocr)

### Performance Options
- `--num-workers`: Number of worker processes (default: 4)
- `--batch-size`: Frame processing batch size (default: 8)
- `--dedup-mode`: Deduplication mode (ssim, text, both)
- `--ssim-threshold`: SSIM similarity threshold (default: 0.80)

## Output Structure

```
output/
├── video_name/
│   ├── frame_000000.jpg
│   ├── frame_000001.jpg
│   ├── ...
│   └── metadata.json
├── processing_summary_scenes.json
└── processing_time_scenes.txt
```

## Scene Detection Benefits

- **95%+ frame reduction**: Process only meaningful keyframes
- **Automatic scene analysis**: No manual intervention required
- **Content coverage**: Ensures representation from all scenes
- **Processing efficiency**: Dramatically faster than frame-by-frame analysis

## Requirements

- Python 3.7+
- OpenCV
- PySceneDetect
- Tesseract OCR (for text extraction)
- See `requirements.txt` for complete list

## Performance

The scene detection mode typically achieves:
- 95%+ reduction in frames processed
- 10-20x faster processing times
- Maintains comprehensive content coverage
- Intelligent keyframe selection

## Troubleshooting

### Tesseract Path Issues
If you encounter Tesseract errors, ensure it's installed and update the path in `video_to_frames.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Docker Usage

This project can be run using Docker to simplify dependency management and ensure a consistent environment.

### Building and Running the Container

To build and run the Docker container, use the following command:

```bash
docker-compose up -d --build
```

This will start a PostgreSQL container, build the necessary image, and populate the database with the images found in the `./output` directory.

### Verifying the Data

You can connect to the PostgreSQL database to verify that the data has been populated correctly:

```bash
docker-compose exec -T postgres psql -U myuser -d mydatabase -c "SELECT * FROM video_frames LIMIT 10;"
```

### Repopulating the Database

If you change the images in the `./output` directory and want to repopulate the database, you need to remove the existing data volume. This can be done with the following command:

```bash
docker-compose down -v
```

After running this command, you can then restart the container to populate the database with the new images:

```bash
docker-compose up -d --build
```

### Memory Issues
For large videos, reduce batch size and number of workers:
```bash
python video_to_frames.py --scene-detection --batch-size 4 --num-workers 2
```
