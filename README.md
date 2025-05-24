# DBSCAN Face Clustering for Dropbox Photos

A Python application that automatically detects and clusters faces in your Dropbox photos using DBSCAN algorithm, without downloading files locally. The system tags photos with metadata about detected faces and clusters.

## Features

- **Stream Processing**: Processes photos directly from Dropbox without local downloads
- **Face Detection**: Detects all faces in photos using face_recognition library
- **Smart Clustering**: Groups similar faces using DBSCAN clustering algorithm
- **Metadata Tagging**: Automatically tags Dropbox files with face cluster information
- **Scalable**: Handles large photo collections efficiently

## Prerequisites

- Python 3.12 or higher
- Dropbox account with API access
- uv package manager

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd dbscan-dropbox
```

2. Install dependencies using uv:
```bash
uv sync
```

## Dropbox Setup

1. Go to [Dropbox App Console](https://www.dropbox.com/developers/apps)
2. Click "Create app"
3. Choose "Scoped access"
4. Select "Full Dropbox" access
5. Name your app and create it
6. In the app settings, generate an access token
7. Copy the access token for use in the configuration

## Configuration

Edit `main.py` and update the following variables in the `main()` function:

```python
DROPBOX_TOKEN = "YOUR_DROPBOX_ACCESS_TOKEN"  # Your Dropbox API token
DROPBOX_FOLDER = "/Photos"  # Path to your photos folder
EPS = 0.6  # DBSCAN distance threshold (lower = tighter clusters)
MIN_SAMPLES = 2  # Minimum faces needed to form a cluster
```

### Parameter Tuning

- **EPS**: Controls how similar faces must be to cluster together (0.3-0.8 typical)
- **MIN_SAMPLES**: Minimum number of similar faces to form a group

## Usage

Run the application:
```bash
uv run python main.py
```

The program will:
1. Scan your specified Dropbox folder for images
2. Detect faces in each image
3. Cluster similar faces across all images
4. Tag files with metadata about detected faces
5. Save results locally in `face_recognition_results.json`

## Output

### Dropbox Metadata
Each processed photo gets tagged with:
- Total number of faces detected
- Cluster IDs for recognized faces
- Count of unique/unrecognized faces

### Local Results File
A JSON file containing:
- Complete clustering results
- File-by-file metadata
- Processing statistics

## Example Output

```
Scanning Dropbox folder: /Photos
Found 150 photos
Processing 1/150: vacation_001.jpg
Found 3 faces in vacation_001.jpg
...
Total faces detected across all images: 423
Clustering 423 faces using DBSCAN...
Found 12 clusters and 35 unique faces
Successfully tagged 150 files, 0 failed
Results saved locally to: face_recognition_results.json
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Performance Notes

- Processing time depends on number and size of images
- API rate limits are handled with automatic delays
- Face detection uses HOG model for speed
- All processing happens in memory

## Troubleshooting

### "Error accessing Dropbox"
- Verify your access token is correct
- Ensure the folder path exists
- Check your Dropbox API permissions

### "No faces detected"
- Ensure photos contain clear, front-facing faces
- Try adjusting the face detection model in code

### High memory usage
- Process folders in smaller batches
- Reduce image resolution before processing

## Privacy & Security

- No images are stored locally
- Face encodings are numerical representations only
- Results file contains metadata, not actual images
- All processing happens in your Dropbox account

## License

[Add your license here]

## Contributing

[Add contribution guidelines if applicable]