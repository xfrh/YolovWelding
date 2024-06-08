import os
import cv2

# Define the input and output directories
input_dir = "D:\\dataset\\images\\train\\collapse"  # Replace with your input folder path
output_dir = "D:\\dataset\images\\train\\collapse\\output"  # Replace with your output folder path

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Process each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png"):
        # Construct the full input path
        input_path = os.path.join(input_dir, filename)

        # Read the image
        image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to 958x653 pixels
        resized_image = cv2.resize(gray_image, (958, 653))

        # Construct the output path
        base_filename = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_filename}.jpg")

        # Save the processed image as a JPG file
        cv2.imwrite(output_path, resized_image)

        print(f"Processed {filename} -> {output_path}")

print("Batch processing completed.")
