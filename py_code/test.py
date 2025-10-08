import pycolmap
import os
from pathlib import Path

def simple_pycolmap_test():
    # Define paths
    image_dir = Path(r"C:/Users/Shadow/Downloads/chaise_1")
    database_path = str(Path("database.db"))  # Convertir en string
    output_path = Path("reconstruction")

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Extract features and match them
    print("Extracting features and matching...")

    # Feature extraction options
    feature_extractor_options = pycolmap.SiftExtractionOptions()

    # Feature matching options
    matcher_options = pycolmap.ExhaustiveMatchingOptions()

    exhaustive_matching_options = pycolmap.ExhaustiveMatchingOptions()
    exhaustive_matching_options.block_size = 50

    # Extract features
    pycolmap.extract_features(
        database_path=database_path,
        image_path=str(image_dir),
        sift_options=feature_extractor_options
    )

    # Match features - utiliser la signature correcte
    pycolmap.match_exhaustive(
        database_path=database_path,
        matching_options=exhaustive_matching_options
    )

    # Mapper options
    mapper_options = pycolmap.IncrementalMapperOptions()
    mapper_options.num_threads = -1  # Use all available CPU cores

    # Run the reconstruction
    print("Running reconstruction...")
    reconstruction = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=str(image_dir),
        output_path=str(output_path),
    )

    if reconstruction is None or len(reconstruction) == 0:
        print("Reconstruction failed!")
        return

    # Print some statistics about the reconstruction
    print(f"Reconstruction:")

    print(f"Reconstruction saved to: {output_path}")

if __name__ == "__main__":
    simple_pycolmap_test()