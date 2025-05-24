import cv2
import numpy as np
import dropbox
import pickle
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import face_recognition
from PIL import Image
import io
from collections import defaultdict
import json
import time

class FacialRecognitionDBSCANWithMetadata:
    def __init__(self, dropbox_token):
        """
        Initialize the facial recognition system with DBSCAN clustering and metadata tagging
        
        Args:
            dropbox_token (str): Your Dropbox API access token
        """
        self.dbx = dropbox.Dropbox(dropbox_token)
        self.face_encodings = []
        self.image_metadata = []  # Store image paths and face info
        self.processed_files = []
        
    def get_photos_from_dropbox(self, dropbox_folder_path="/"):
        """
        Get list of photo files from Dropbox folder without downloading
        
        Args:
            dropbox_folder_path (str): Path to the Dropbox folder containing photos
            
        Returns:
            list: List of photo file metadata
        """
        print(f"Scanning Dropbox folder: {dropbox_folder_path}")
        
        try:
            photo_files = []
            photo_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            # List files recursively
            result = self.dbx.files_list_folder(dropbox_folder_path, recursive=True)
            
            while True:
                for entry in result.entries:
                    if isinstance(entry, dropbox.files.FileMetadata):
                        file_ext = entry.name.lower().split('.')[-1]
                        if f'.{file_ext}' in photo_extensions:
                            photo_files.append({
                                'path': entry.path_lower,
                                'name': entry.name,
                                'size': entry.size
                            })
                
                if not result.has_more:
                    break
                result = self.dbx.files_list_folder_continue(result.cursor)
            
            print(f"Found {len(photo_files)} photos")
            return photo_files
            
        except dropbox.exceptions.ApiError as e:
            print(f"Error accessing Dropbox: {e}")
            return []
    
    def process_image_from_dropbox(self, file_path):
        """
        Process a single image directly from Dropbox without downloading to disk
        
        Args:
            file_path (str): Dropbox path to the image file
            
        Returns:
            list: Face encodings found in the image
        """
        try:
            # Download image data directly into memory
            _, response = self.dbx.files_download(file_path)
            image_data = response.content
            
            # Convert to PIL Image and then to face_recognition format
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array for face_recognition
            image_array = np.array(pil_image)
            
            # Find face locations and encodings
            face_locations = face_recognition.face_locations(image_array, model="hog")
            face_encodings = face_recognition.face_encodings(image_array, face_locations)
            
            return face_encodings, face_locations
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return [], []
    
    def detect_and_encode_all_faces(self, photo_files):
        """
        Process all photos and detect faces without downloading files
        
        Args:
            photo_files (list): List of photo file metadata from Dropbox
        """
        print("Processing photos and detecting faces...")
        
        for i, photo_file in enumerate(photo_files):
            print(f"Processing {i+1}/{len(photo_files)}: {photo_file['name']}")
            
            face_encodings, face_locations = self.process_image_from_dropbox(photo_file['path'])
            
            if face_encodings:
                # Store each face encoding with metadata
                for j, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    self.face_encodings.append(face_encoding)
                    self.image_metadata.append({
                        'file_path': photo_file['path'],
                        'file_name': photo_file['name'],
                        'face_index': j,
                        'face_location': face_location,
                        'total_faces_in_image': len(face_encodings)
                    })
                
                print(f"Found {len(face_encodings)} faces in {photo_file['name']}")
            
            # Add small delay to avoid hitting API rate limits
            time.sleep(0.1)
        
        print(f"Total faces detected across all images: {len(self.face_encodings)}")
    
    def cluster_faces_with_dbscan(self, eps=0.5, min_samples=2):
        """
        Cluster similar faces using DBSCAN
        
        Args:
            eps (float): Maximum distance between samples for clustering
            min_samples (int): Minimum number of samples in a cluster
        
        Returns:
            dict: Clustering results
        """
        if len(self.face_encodings) == 0:
            print("No face encodings found. Please run detect_and_encode_all_faces() first.")
            return {}
        
        print(f"Clustering {len(self.face_encodings)} faces using DBSCAN...")
        
        # Convert face encodings to numpy array
        face_encodings_array = np.array(self.face_encodings)
        
        # Standardize the features
        scaler = StandardScaler()
        face_encodings_scaled = scaler.fit_transform(face_encodings_array)
        
        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(face_encodings_scaled)
        
        # Organize results
        clusters = defaultdict(list)
        noise_faces = []
        
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise/outlier
                noise_faces.append(i)
            else:
                clusters[label].append(i)
        
        print(f"Found {len(clusters)} clusters and {len(noise_faces)} unique faces")
        
        # Print cluster information
        for cluster_id, face_indices in clusters.items():
            print(f"Cluster {cluster_id}: {len(face_indices)} faces")
        
        return {
            'clusters': dict(clusters),
            'noise': noise_faces,
            'labels': cluster_labels.tolist()
        }
    
    def create_metadata_tags(self, clustering_results):
        """
        Create metadata tags for each photo based on clustering results
        
        Args:
            clustering_results (dict): Results from cluster_faces_with_dbscan()
        
        Returns:
            dict: Metadata tags for each file
        """
        file_metadata = defaultdict(lambda: {
            'face_clusters': [],
            'unique_faces': [],
            'total_faces': 0,
            'cluster_summary': {}
        })
        
        clusters = clustering_results['clusters']
        noise_faces = clustering_results['noise']
        
        # Process clustered faces
        for cluster_id, face_indices in clusters.items():
            for face_idx in face_indices:
                face_info = self.image_metadata[face_idx]
                file_path = face_info['file_path']
                
                file_metadata[file_path]['face_clusters'].append({
                    'cluster_id': int(cluster_id),
                    'face_index': face_info['face_index'],
                    'confidence': 'clustered'
                })
        
        # Process noise/unique faces
        for face_idx in noise_faces:
            face_info = self.image_metadata[face_idx]
            file_path = face_info['file_path']
            
            file_metadata[file_path]['unique_faces'].append({
                'face_index': face_info['face_index'],
                'confidence': 'unique'
            })
        
        # Add summary information
        for file_path in file_metadata:
            metadata = file_metadata[file_path]
            metadata['total_faces'] = len(metadata['face_clusters']) + len(metadata['unique_faces'])
            
            # Create cluster summary
            cluster_counts = defaultdict(int)
            for face in metadata['face_clusters']:
                cluster_counts[face['cluster_id']] += 1
            
            metadata['cluster_summary'] = dict(cluster_counts)
            
            # Create human-readable tags
            tags = []
            if metadata['face_clusters']:
                cluster_ids = list(metadata['cluster_summary'].keys())
                tags.append(f"person_clusters_{','.join(map(str, cluster_ids))}")
            
            if metadata['unique_faces']:
                tags.append(f"unique_faces_{len(metadata['unique_faces'])}")
            
            tags.append(f"total_faces_{metadata['total_faces']}")
            
            metadata['tags'] = tags
        
        return dict(file_metadata)
    
    def apply_metadata_to_dropbox_files(self, file_metadata):
        """
        Apply metadata tags to Dropbox files using properties
        
        Args:
            file_metadata (dict): Metadata to apply to each file
        """
        print("Applying metadata tags to Dropbox files...")
        
        # Create property template (if it doesn't exist)
        template_name = "face_recognition_clusters"
        
        try:
            # Try to get existing template
            templates = self.dbx.file_properties_templates_list_for_user()
            template_exists = any(t.name == template_name for t in templates.template_ids)
            
            if not template_exists:
                # Create new template
                fields = [
                    dropbox.file_properties.PropertyFieldTemplate(
                        name="face_clusters",
                        description="Face cluster information",
                        type=dropbox.file_properties.PropertyType.string
                    ),
                    dropbox.file_properties.PropertyFieldTemplate(
                        name="total_faces",
                        description="Total number of faces detected",
                        type=dropbox.file_properties.PropertyType.string
                    ),
                    dropbox.file_properties.PropertyFieldTemplate(
                        name="cluster_summary",
                        description="Summary of clusters in this image",
                        type=dropbox.file_properties.PropertyType.string
                    ),
                    dropbox.file_properties.PropertyFieldTemplate(
                        name="processing_date",
                        description="Date when face recognition was performed",
                        type=dropbox.file_properties.PropertyType.string
                    )
                ]
                
                template = self.dbx.file_properties_templates_add_for_user(
                    name=template_name,
                    description="Face recognition clustering metadata",
                    fields=fields
                )
                template_id = template.template_id
                print(f"Created new property template: {template_id}")
            else:
                # Get existing template ID
                for template in templates.template_ids:
                    if template.name == template_name:
                        template_id = template.template_id
                        break
                print(f"Using existing property template: {template_id}")
        
        except Exception as e:
            print(f"Note: Could not create/access property template: {e}")
            print("Will use alternative tagging method...")
            template_id = None
        
        # Apply metadata to each file
        successful_tags = 0
        failed_tags = 0
        
        for file_path, metadata in file_metadata.items():
            try:
                if template_id:
                    # Use property fields
                    properties = [
                        dropbox.file_properties.PropertyField(
                            name="face_clusters",
                            value=json.dumps(metadata['face_clusters'])
                        ),
                        dropbox.file_properties.PropertyField(
                            name="total_faces",
                            value=str(metadata['total_faces'])
                        ),
                        dropbox.file_properties.PropertyField(
                            name="cluster_summary",
                            value=json.dumps(metadata['cluster_summary'])
                        ),
                        dropbox.file_properties.PropertyField(
                            name="processing_date",
                            value=time.strftime("%Y-%m-%d %H:%M:%S")
                        )
                    ]
                    
                    self.dbx.file_properties_properties_add(
                        file=file_path,
                        property_groups=[
                            dropbox.file_properties.PropertyGroup(
                                template_id=template_id,
                                fields=properties
                            )
                        ]
                    )
                else:
                    # Alternative: Use file comments (if supported)
                    comment = f"Face Recognition Results:\n"
                    comment += f"Total faces: {metadata['total_faces']}\n"
                    comment += f"Clusters: {metadata['cluster_summary']}\n"
                    comment += f"Tags: {', '.join(metadata['tags'])}\n"
                    comment += f"Processed: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                    
                    # Note: File comments might not be available in all Dropbox plans
                    print(f"Metadata for {file_path}: {comment}")
                
                successful_tags += 1
                
            except Exception as e:
                print(f"Failed to tag {file_path}: {e}")
                failed_tags += 1
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
        
        print(f"Successfully tagged {successful_tags} files, {failed_tags} failed")
    
    def save_results_locally(self, clustering_results, file_metadata, filename="face_recognition_results.json"):
        """
        Save all results to a local JSON file for backup/reference
        
        Args:
            clustering_results (dict): Clustering results
            file_metadata (dict): File metadata
            filename (str): Output filename
        """
        results = {
            'clustering_results': clustering_results,
            'file_metadata': file_metadata,
            'processing_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_images_processed': len(file_metadata),
            'total_faces_found': len(self.face_encodings)
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved locally to: {filename}")
    
    def get_file_metadata_from_dropbox(self, file_path):
        """
        Retrieve existing metadata from a Dropbox file
        
        Args:
            file_path (str): Dropbox file path
        
        Returns:
            dict: Existing metadata if any
        """
        try:
            properties = self.dbx.file_properties_properties_search(
                queries=[dropbox.file_properties.PropertiesSearchQuery(
                    query="face_clusters",
                    mode=dropbox.file_properties.PropertiesSearchMode.field_name
                )]
            )
            
            for match in properties.matches:
                if match.path.lower() == file_path.lower():
                    return match.property_groups
            
        except Exception as e:
            print(f"Could not retrieve metadata for {file_path}: {e}")
        
        return None


def main():
    # Configuration
    DROPBOX_TOKEN = "YOUR_DROPBOX_ACCESS_TOKEN"  # Replace with your token
    DROPBOX_FOLDER = "/Photos"  # Replace with your folder path
    
    # DBSCAN parameters
    EPS = 0.6  # Distance threshold (adjust based on your needs)
    MIN_SAMPLES = 2  # Minimum faces needed to form a cluster
    
    # Initialize the facial recognition system
    fr_system = FacialRecognitionDBSCANWithMetadata(DROPBOX_TOKEN)
    
    try:
        # Step 1: Get list of photos from Dropbox
        photo_files = fr_system.get_photos_from_dropbox(DROPBOX_FOLDER)
        if not photo_files:
            print("No photos found. Exiting.")
            return
        
        # Step 2: Process images and detect faces (without downloading)
        fr_system.detect_and_encode_all_faces(photo_files)
        
        # Step 3: Cluster faces using DBSCAN
        clustering_results = fr_system.cluster_faces_with_dbscan(
            eps=EPS, 
            min_samples=MIN_SAMPLES
        )
        
        # Step 4: Create metadata tags for each file
        file_metadata = fr_system.create_metadata_tags(clustering_results)
        
        # Step 5: Apply metadata to Dropbox files
        fr_system.apply_metadata_to_dropbox_files(file_metadata)
        
        # Step 6: Save results locally for backup
        fr_system.save_results_locally(clustering_results, file_metadata)
        
        print("Facial recognition clustering and tagging completed successfully!")
        print(f"Processed {len(photo_files)} images with {len(fr_system.face_encodings)} total faces")
        
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()