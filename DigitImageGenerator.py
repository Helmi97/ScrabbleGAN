import os
import random
import rarfile
import gdown
from PIL import Image, ImageFilter
import numpy as np

class DigitImageGenerator:
    def __init__(self, dataset_path="dida_dataset\\10000"):
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            self.download_dida_dataset()

    # Step 1: Download the DIDA dataset
    def download_dida_dataset(self):
        url = "https://drive.google.com/uc?id=1d-U-lxIoS5QuPEYPvHA2-Bm4pULsTb06"
        output = "dida_dataset.rar"
        
        # Only download if the file does not exist
        if not os.path.exists(output):
            gdown.download(url, output, quiet=False)

        # Check if the downloaded file is a valid RAR file
        if not rarfile.is_rarfile(output):
            raise rarfile.BadRarFile("The downloaded file is not a valid RAR file. Please check the download link.")
        
        # Extract all files from the RAR archive
        with rarfile.RarFile(output, 'r') as rar_ref:
            rar_ref.extractall(self.dataset_path)

        # Adjust the dataset_path to point to the correct folder containing digit folders
        extracted_items = os.listdir(self.dataset_path)
        for item in extracted_items:
            item_path = os.path.join(self.dataset_path, item)
            if os.path.isdir(item_path) and "10000" in item:
                self.dataset_path = item_path
                break

    # Step 2: For each digit, randomly select an image from the dataset
    def select_images_for_digits(self, digits_str):
        selected_images = []
        for digit in digits_str:
            digit_path = os.path.join(self.dataset_path, digit)
            if not os.path.exists(digit_path):
                raise FileNotFoundError(f"The directory for digit '{digit}' was not found in the dataset path '{self.dataset_path}'.")
            image_files = [f for f in os.listdir(digit_path) if os.path.isfile(os.path.join(digit_path, f))]
            if not image_files:
                raise FileNotFoundError(f"No images found for digit '{digit}' in path '{digit_path}'.")
            selected_image = random.choice(image_files)
            selected_images.append(os.path.join(digit_path, selected_image))
        return selected_images

    # Step 3: Stitch images together with padding
    def stitch_images(self, image_paths, padding):
        images = [Image.open(image_path) for image_path in image_paths]
        min_height = min(img.height for img in images)
        resized_images = [img.resize((int(img.width * (min_height / img.height)), min_height), Image.LANCZOS) for img in images]
        widths, heights = zip(*(img.size for img in resized_images))
        total_width = sum(widths) + padding * (len(resized_images) - 1)
        max_height = min_height

        stitched_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))
        x_offset = 0
        for img in resized_images:
            stitched_image.paste(img, (x_offset, 0))
            x_offset += img.width + padding

        return stitched_image

    # Create the final image given a digit string and padding
    def create_digits_image(self, digit_string, padding=0):
        # Select images for each digit
        selected_image_paths = self.select_images_for_digits(digit_string)

        # Stitch the images together
        stitched_image = self.stitch_images(selected_image_paths, padding)

        # Convert to black and white
        bw_image = self.convert_to_black_and_white(stitched_image)
        
        # Remove noise from the black and white image
        cleaned_image = self.remove_noise(bw_image)

        # Convert to NumPy array and return
        return np.array(cleaned_image)

    # Convert images to black and white based on a threshold
    def convert_to_black_and_white(self, image, threshold=128):
        bw_image = image.convert('L')  # Convert to grayscale
        bw_image = bw_image.point(lambda x: 0 if x < threshold else 255, '1')  # Apply threshold
        return bw_image

    # Remove small black bodies to reduce noise
    def remove_noise(self, image, filter_size=3):
        # Apply a median filter to reduce noise
        filtered_image = image.filter(ImageFilter.MedianFilter(filter_size))
        return filtered_image

if __name__ == "__main__":
    # User inputs
    digits_str = "12345"  # Example input string of digits

    # Create an instance of DigitImageGenerator
    generator = DigitImageGenerator()

    # Create the final image
    stitched_image = generator.create_digits_image(digits_str)
    
    #to bw
    bw_image= generator.convert_to_black_and_white(stitched_image, threshold=128)

    # Remove noise from the black and white image
    cleaned_image = generator.remove_noise(bw_image)

    # Save and show the cleaned black and white image
    cleaned_image.save("stitched_digits_bw_cleaned.png")
    stitched_image.show()
    bw_image.show()
    cleaned_image.show()