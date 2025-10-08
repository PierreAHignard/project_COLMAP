import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pycolmap
import threading
import numpy as np
from pathlib import Path

class ColmapGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("COLMAP 3D Reconstruction")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create title
        title_label = ttk.Label(self.main_frame, text="COLMAP 3D Reconstruction", font=("Arial", 18, "bold"))
        title_label.pack(pady=10)

        # Create input frame
        input_frame = ttk.LabelFrame(self.main_frame, text="Input/Output Settings", padding="10")
        input_frame.pack(fill=tk.X, pady=10)

        # Image directory
        ttk.Label(input_frame, text="Image Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.image_dir_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.image_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_image_dir).grid(row=0, column=2, padx=5, pady=5)

        # Output directory
        ttk.Label(input_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_dir_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_output_dir).grid(row=1, column=2, padx=5, pady=5)

        # Create options frame
        options_frame = ttk.LabelFrame(self.main_frame, text="Reconstruction Options", padding="10")
        options_frame.pack(fill=tk.X, pady=10)

        # Max image size
        ttk.Label(options_frame, text="Max Image Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.max_image_size_var = tk.IntVar(value=2000)
        ttk.Entry(options_frame, textvariable=self.max_image_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Max features
        ttk.Label(options_frame, text="Max Features per Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.max_features_var = tk.IntVar(value=8000)
        ttk.Entry(options_frame, textvariable=self.max_features_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Match method
        ttk.Label(options_frame, text="Matching Method:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.match_method_var = tk.StringVar(value="exhaustive")
        match_methods = ["exhaustive", "sequential", "vocab_tree", "spatial"]
        ttk.Combobox(options_frame, textvariable=self.match_method_var, values=match_methods, width=15).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Dense reconstruction
        self.dense_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Perform Dense Reconstruction", variable=self.dense_var).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Use GPU
        self.use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Use GPU (if available)", variable=self.use_gpu_var).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Advanced options
        advanced_button = ttk.Button(options_frame, text="Advanced Options...", command=self.show_advanced_options)
        advanced_button.grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Create log frame
        log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Log text
        self.log_text = tk.Text(log_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for log
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=10)

        # Buttons frame
        buttons_frame = ttk.Frame(self.main_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        # Start button
        self.start_button = ttk.Button(buttons_frame, text="Start Reconstruction", command=self.start_reconstruction)
        self.start_button.pack(side=tk.RIGHT, padx=5)

        # Cancel button
        self.cancel_button = ttk.Button(buttons_frame, text="Cancel", command=self.cancel_reconstruction, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.RIGHT, padx=5)

        # Initialize advanced options
        self.advanced_options = {
            "min_num_matches": 15,
            "mapper_min_model_size": 10,
            "mapper_max_extra_param": 1.0,
            "patch_match_window_radius": 5,
            "patch_match_window_step": 1,
            "fusion_min_num_pixels": 5,
            "meshing_trim": 7
        }

        # Initialize reconstruction thread
        self.reconstruction_thread = None
        self.cancel_flag = False

        # Redirect stdout to log
        self.redirect_stdout()

        # Log initial message
        self.log("COLMAP 3D Reconstruction GUI initialized")
        self.log("Please select an image directory and output directory to begin")

    def redirect_stdout(self):
        """Redirect stdout to the log text widget"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget

            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)

            def flush(self):
                pass

        sys.stdout = StdoutRedirector(self.log_text)

    def log(self, message):
        """Add a message to the log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def browse_image_dir(self):
        """Browse for image directory"""
        directory = filedialog.askdirectory(title="Select Image Directory")
        if directory:
            self.image_dir_var.set(directory)

    def browse_output_dir(self):
        """Browse for output directory"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_var.set(directory)

    def show_advanced_options(self):
        """Show advanced options dialog"""
        advanced_window = tk.Toplevel(self.root)
        advanced_window.title("Advanced Options")
        advanced_window.geometry("500x400")
        advanced_window.transient(self.root)
        advanced_window.grab_set()

        frame = ttk.Frame(advanced_window, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)

        # Feature matching options
        match_frame = ttk.LabelFrame(frame, text="Feature Matching", padding="10")
        match_frame.pack(fill=tk.X, pady=5)

        ttk.Label(match_frame, text="Min Number of Matches:").grid(row=0, column=0, sticky=tk.W, pady=5)
        min_matches_var = tk.IntVar(value=self.advanced_options["min_num_matches"])
        ttk.Entry(match_frame, textvariable=min_matches_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Mapper options
        mapper_frame = ttk.LabelFrame(frame, text="Mapper Options", padding="10")
        mapper_frame.pack(fill=tk.X, pady=5)

        ttk.Label(mapper_frame, text="Min Model Size:").grid(row=0, column=0, sticky=tk.W, pady=5)
        min_model_size_var = tk.IntVar(value=self.advanced_options["mapper_min_model_size"])
        ttk.Entry(mapper_frame, textvariable=min_model_size_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(mapper_frame, text="Max Extra Param:").grid(row=1, column=0, sticky=tk.W, pady=5)
        max_extra_param_var = tk.DoubleVar(value=self.advanced_options["mapper_max_extra_param"])
        ttk.Entry(mapper_frame, textvariable=max_extra_param_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Dense reconstruction options
        dense_frame = ttk.LabelFrame(frame, text="Dense Reconstruction", padding="10")
        dense_frame.pack(fill=tk.X, pady=5)

        ttk.Label(dense_frame, text="Patch Match Window Radius:").grid(row=0, column=0, sticky=tk.W, pady=5)
        window_radius_var = tk.IntVar(value=self.advanced_options["patch_match_window_radius"])
        ttk.Entry(dense_frame, textvariable=window_radius_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(dense_frame, text="Patch Match Window Step:").grid(row=1, column=0, sticky=tk.W, pady=5)
        window_step_var = tk.IntVar(value=self.advanced_options["patch_match_window_step"])
        ttk.Entry(dense_frame, textvariable=window_step_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(dense_frame, text="Fusion Min Num Pixels:").grid(row=2, column=0, sticky=tk.W, pady=5)
        min_pixels_var = tk.IntVar(value=self.advanced_options["fusion_min_num_pixels"])
        ttk.Entry(dense_frame, textvariable=min_pixels_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(dense_frame, text="Meshing Trim:").grid(row=3, column=0, sticky=tk.W, pady=5)
        meshing_trim_var = tk.IntVar(value=self.advanced_options["meshing_trim"])
        ttk.Entry(dense_frame, textvariable=meshing_trim_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)

        # Buttons
        buttons_frame = ttk.Frame(frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        def save_options():
            self.advanced_options["min_num_matches"] = min_matches_var.get()
            self.advanced_options["mapper_min_model_size"] = min_model_size_var.get()
            self.advanced_options["mapper_max_extra_param"] = max_extra_param_var.get()
            self.advanced_options["patch_match_window_radius"] = window_radius_var.get()
            self.advanced_options["patch_match_window_step"] = window_step_var.get()
            self.advanced_options["fusion_min_num_pixels"] = min_pixels_var.get()
            self.advanced_options["meshing_trim"] = meshing_trim_var.get()
            advanced_window.destroy()

        ttk.Button(buttons_frame, text="Save", command=save_options).pack(side=tk.RIGHT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=advanced_window.destroy).pack(side=tk.RIGHT, padx=5)

    def start_reconstruction(self):
        """Start the reconstruction process"""
        # Validate inputs
        image_dir = self.image_dir_var.get()
        output_dir = self.output_dir_var.get()

        if not image_dir or not os.path.isdir(image_dir):
            messagebox.showerror("Error", "Please select a valid image directory")
            return

        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory")
            return

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Disable start button and enable cancel button
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)

        # Reset progress and log
        self.progress_var.set(0)
        self.log_text.delete(1.0, tk.END)
        self.cancel_flag = False

        # Start reconstruction in a separate thread
        self.reconstruction_thread = threading.Thread(
            target=self.run_reconstruction,
            args=(
                image_dir,
                output_dir,
                self.max_image_size_var.get(),
                self.max_features_var.get(),
                self.match_method_var.get(),
                self.dense_var.get(),
                self.use_gpu_var.get()
            )
        )
        self.reconstruction_thread.daemon = True
        self.reconstruction_thread.start()

    def cancel_reconstruction(self):
        """Cancel the reconstruction process"""
        if messagebox.askyesno("Cancel", "Are you sure you want to cancel the reconstruction?"):
            self.cancel_flag = True
            self.log("Cancelling reconstruction...")

    def run_reconstruction(self, image_dir, output_dir, max_image_size, max_features, match_method, dense, use_gpu):
        """Run the reconstruction process"""
        try:
            # Set up database path
            database_path = os.path.join(output_dir, "database.db")

            # Create timer to measure performance
            timer = pycolmap.Timer()
            timer.start()

            # Update progress
            self.update_progress(5, "Setting up reconstruction...")

            # Step 1: Feature extraction
            self.log(f"Extracting features from images in {image_dir}")

            # Configure feature extraction
            sift_options = pycolmap.SiftExtractionOptions()
            sift_options.max_num_features = max_features
            sift_options.use_gpu = use_gpu

            # Configure image reader
            image_reader_options = pycolmap.ImageReaderOptions()
            image_reader_options.camera_model = "RADIAL"
            image_reader_options.single_camera_per_folder = False
            image_reader_options.default_focal_length_factor = 1.2
            image_reader_options.max_image_size = max_image_size

            # Extract features
            if self.cancel_flag:
                self.finish_reconstruction(False)
                return

            pycolmap.extract_features(
                database_path=database_path,
                image_path=image_dir,
                sift_options=sift_options,
                image_reader_options=image_reader_options
            )

            # Update progress
            self.update_progress(20, "Feature extraction completed")

            # Step 2: Feature matching
            self.log("Matching features between images")

            if match_method == "exhaustive":
                # Exhaustive matching (compare all image pairs)
                match_options = pycolmap.ExhaustiveMatchingOptions()
                pycolmap.match_exhaustive(
                    database_path=database_path,
                    match_options=match_options
                )
            elif match_method == "sequential":
                # Sequential matching (for ordered image sequences)
                match_options = pycolmap.SequentialMatchingOptions()
                match_options.overlap = 10
                pycolmap.match_sequential(
                    database_path=database_path,
                    match_options=match_options
                )
            elif match_method == "vocab_tree":
                # Vocabulary tree matching (faster for large image sets)
                match_options = pycolmap.VocabTreeMatchingOptions()
                match_options.num_images = 50

                # Check if vocab tree file exists
                vocab_tree_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab_tree.bin")
                if not os.path.exists(vocab_tree_path):
                    self.log("Warning: vocab_tree.bin not found. Downloading...")
                    # You would need to implement downloading the vocab tree file here
                    # For now, fallback to exhaustive matching
                    match_options = pycolmap.ExhaustiveMatchingOptions()
                    pycolmap.match_exhaustive(
                        database_path=database_path,
                        match_options=match_options
                    )
                else:
                    pycolmap.match_vocabtree(
                        database_path=database_path,
                        match_options=match_options,
                        vocab_tree_path=vocab_tree_path
                    )
            elif match_method == "spatial":
                # Spatial matching (for images with GPS data)
                match_options = pycolmap.SpatialMatchingOptions()
                match_options.max_num_neighbors = 50
                pycolmap.match_spatial(
                    database_path=database_path,
                    match_options=match_options
                )

            # Verify matches to filter outliers
            if self.cancel_flag:
                self.finish_reconstruction(False)
                return

            pycolmap.verify_matches(database_path)

            # Update progress
            self.update_progress(40, "Feature matching completed")

            # Step 3: Sparse reconstruction (Structure from Motion)
            self.log("Performing incremental mapping (sparse reconstruction)")

            # Configure mapper
            mapper_options = pycolmap.IncrementalMapperOptions()
            mapper_options.min_model_size = self.advanced_options["mapper_min_model_size"]
            mapper_options.max_extra_param = self.advanced_options["mapper_max_extra_param"]

            # Create reconstruction manager
            reconstruction_manager = pycolmap.ReconstructionManager()

            # Run incremental mapping
            sparse_dir = os.path.join(output_dir, "sparse")
            os.makedirs(sparse_dir, exist_ok=True)

            if self.cancel_flag:
                self.finish_reconstruction(False)
                return

            pycolmap.incremental_mapping(
                database_path=database_path,
                image_path=image_dir,
                output_path=sparse_dir,
                reconstruction_manager=reconstruction_manager,
                mapper_options=mapper_options
            )

            if reconstruction_manager.size() == 0:
                self.log("Reconstruction failed. No models were created.")
                self.finish_reconstruction(False)
                return

            # Get the largest reconstruction
            reconstruction = reconstruction_manager.get(0)
            self.log(f"Sparse reconstruction completed with {reconstruction.num_points3D()} 3D points")

            # Update progress
            self.update_progress(70, "Sparse reconstruction completed")

            # Step 4: Dense reconstruction (if requested)
            if dense:
                dense_dir = os.path.join(output_dir, "dense")
                os.makedirs(dense_dir, exist_ok=True)

                # Step 4.1: Undistort images
                self.log("Undistorting images")
                undistort_options = pycolmap.UndistortCameraOptions()
                undistort_options.max_image_size = max_image_size

                if self.cancel_flag:
                    self.finish_reconstruction(False)
                    return

                pycolmap.undistort_images(
                    output_path=dense_dir,
                    image_path=image_dir,
                    reconstruction=reconstruction,
                    options=undistort_options
                )

                # Update progress
                self.update_progress(80, "Images undistorted")

                # Step 4.2: Patch match stereo
                self.log("Running patch match stereo")
                patch_match_options = pycolmap.PatchMatchOptions()
                patch_match_options.gpu_index = "0" if use_gpu else "-1"
                patch_match_options.window_radius = self.advanced_options["patch_match_window_radius"]
                patch_match_options.window_step = self.advanced_options["patch_match_window_step"]

                if self.cancel_flag:
                    self.finish_reconstruction(False)
                    return

                pycolmap.patch_match_stereo(
                    workspace_path=dense_dir,
                    options=patch_match_options
                )

                # Update progress
                self.update_progress(90, "Patch match stereo completed")

                # Step 4.3: Stereo fusion
                self.log("Performing stereo fusion")
                fusion_options = pycolmap.StereoFusionOptions()
                fusion_options.min_num_pixels = self.advanced_options["fusion_min_num_pixels"]

                if self.cancel_flag:
                    self.finish_reconstruction(False)
                    return

                pycolmap.stereo_fusion(
                    workspace_path=dense_dir,
                    output_path=os.path.join(dense_dir, "fused.ply"),
                    options=fusion_options
                )

                # Step 4.4: Meshing (create a textured mesh)
                self.log("Creating mesh from point cloud")
                meshing_options = pycolmap.PoissonMeshingOptions()
                meshing_options.trim = self.advanced_options["meshing_trim"]

                if self.cancel_flag:
                    self.finish_reconstruction(False)
                    return

                pycolmap.poisson_meshing(
                    input_path=os.path.join(dense_dir, "fused.ply"),
                    output_path=os.path.join(dense_dir, "meshed.ply"),
                    options=meshing_options
                )

            # Stop timer and print elapsed time
            timer.pause()
            self.log(f"Total execution time: {timer.elapsed_seconds():.2f} seconds")

            # Update progress
            self.update_progress(100, "Reconstruction completed successfully")

            # Show success message
            self.finish_reconstruction(True)

        except Exception as e:
            self.log(f"Error: {str(e)}")
            self.finish_reconstruction(False)

    def update_progress(self, value, message):
        """Update progress bar and log message"""
        self.progress_var.set(value)
        self.log(message)

    def finish_reconstruction(self, success):
        """Finish the reconstruction process"""
        # Enable start button and disable cancel button
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)

        if success:
            messagebox.showinfo("Success", "Reconstruction completed successfully!")
        elif self.cancel_flag:
            messagebox.showinfo("Cancelled", "Reconstruction was cancelled by the user.")
        else:
            messagebox.showerror("Error", "Reconstruction failed. Check the log for details.")

def main():
    # Create root window
    root = tk.Tk()

    # Create GUI
    app = ColmapGUI(root)

    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()
