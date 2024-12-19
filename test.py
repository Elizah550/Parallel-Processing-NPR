# Import required libraries
import cv2
import time
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# Dataset path
dataset_path = "images_final"

# Helper function to load images from the dataset
def load_images_from_path(path):
    """
    Load images from the specified dataset path.
    """
    images = []
    for file_name in os.listdir(path):
        if file_name.endswith((".jpg", ".png")):
            img = cv2.imread(os.path.join(path, file_name))
            if img is not None:
                images.append((file_name, img))
    return images

# Detection function using OpenCV Haar Cascade
def detect_plate(image):
    """
    Detect license plates in an image using Haar Cascade.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
    detected = plates.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
    return detected

# Serial implementation
def serial_detection(images):
    """
    Perform serial license plate detection.
    """
    results = []
    for _, img in images:
        results.append(detect_plate(img))
    return results

# Parallel implementation with multiprocessing
def parallel_detection(image_data):
    """
    Parallel license plate detection for a single image.
    """
    _, img = image_data
    return detect_plate(img)

def run_parallel_processing(images):
    """
    Run parallel processing for plate detection using multiprocessing.
    """
    num_processes = cpu_count()  # Use all available CPU cores
    with Pool(processes=num_processes) as pool:
        results = pool.map(parallel_detection, images)
    return results

# Measure performance
def measure_performance(images):
    """
    Measure the execution time for serial and parallel implementations.
    """
    # Serial processing
    print("Running serial processing...")
    start_time = time.time()
    serial_results = serial_detection(images)
    serial_time = time.time() - start_time

    # Parallel processing
    print("Running parallel processing...")
    start_time = time.time()
    parallel_results = run_parallel_processing(images)
    parallel_time = time.time() - start_time

    # Speedup and Efficiency
    speedup = serial_time / parallel_time
    efficiency = speedup / cpu_count()

    return serial_time, parallel_time, speedup, efficiency, serial_results, parallel_results

# Function to save and display detected regions
def save_and_show_detected_regions(images, results, output_folder="output"):
    """
    Save cropped license plate regions and display images with red rectangles.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    count = 0
    detection_counts = []
    file_names = []

    for i, ((file_name, img), detections) in enumerate(zip(images, results)):
        detection_counts.append(len(detections))
        file_names.append(file_name)

        if len(detections) > 0 and count < 5:  # Display and save top 5 detected images
            img_with_rect = img.copy()
            for (x, y, w, h) in detections:
                # Draw red rectangle around detected plate
                cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cropped_plate = img[y:y+h, x:x+w]

                # Save cropped region and image with rectangles
                cv2.imwrite(os.path.join(output_folder, f"cropped_plate_{count}.jpg"), cropped_plate)
                cv2.imwrite(os.path.join(output_folder, f"image_with_rect_{count}.jpg"), img_with_rect)

                # Display the image with rectangles
                plt.figure(figsize=(8, 5))
                plt.imshow(cv2.cvtColor(img_with_rect, cv2.COLOR_BGR2RGB))
                plt.title(f"Detected License Plate in {file_name}")
                plt.axis('off')
                plt.show()

                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break

    # Visualization: Detection Count per Image
    plt.figure(figsize=(10, 6))
    plt.bar(file_names, detection_counts, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.title("Number of License Plates Detected Per Image")
    plt.ylabel("Detections")
    plt.xlabel("Image Name")
    plt.tight_layout()
    plt.show()

# Function to plot execution time comparison
def plot_execution_time(serial_time, parallel_time):
    """
    Plot execution time comparison for Serial and Parallel methods.
    """
    methods = ['Serial', 'Parallel']
    times = [serial_time, parallel_time]

    plt.figure(figsize=(8, 5))
    plt.plot(methods, times, marker='o', linestyle='-', color='blue')
    plt.title('Execution Time Comparison: Serial vs Parallel')
    plt.ylabel('Time (seconds)')
    plt.xlabel('Method')
    plt.grid(True)
    plt.show()

# Main code execution
if __name__ == "__main__":
    print("Loading images...")
    images = load_images_from_path(dataset_path)

    print(f"Loaded {len(images)} images for processing.")
    if not images:
        print("No images found in the dataset path. Ensure dataset is downloaded correctly.")
    else:
        # Measure performance
        serial_time, parallel_time, speedup, efficiency, serial_results, parallel_results = measure_performance(images)

        # Print Results
        print(f"Serial Execution Time: {serial_time:.2f} seconds")
        print(f"Parallel Execution Time: {parallel_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}")
        print(f"Efficiency: {efficiency:.2f}")

        # Visualization: Execution Time Comparison
        plot_execution_time(serial_time, parallel_time)

        # Visualization: Performance Metrics
        metrics = ['Speedup', 'Efficiency']
        values = [speedup, efficiency]

        plt.figure(figsize=(8, 5))
        plt.bar(metrics, values, color=['orange', 'purple'])
        plt.title('Performance Metrics')
        plt.ylabel('Value')
        plt.xlabel('Metric')
        plt.tight_layout()
        plt.show()

        # Save and show detected regions
        print("Processing and saving cropped license plate images...")
        save_and_show_detected_regions(images, serial_results)
        print("Processing completed. Check the 'output' folder for saved images.")
