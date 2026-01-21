import cv2
import os
import numpy as np
from pathlib import Path

class ChestXrayProcessor:
    def __init__(self, target_size=(227, 227), mask_center = True, mask_size = 50):
        self.target_size = target_size
        self.mask_center = mask_center
        self.mask_size = mask_size
    
    def process_single_image(self, image_path, output_path = None):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Image at path {image_path} could not be read.")
            return None

        h, w = img.shape
        scale = max(self.target_size[0] / w, self.target_size[1] / h)
        new_h, new_w = int(h * scale), int(w * scale)

        if new_h < self.target_size[1] or new_w < self.target_size[0]:
            new_h = max(new_h, self.target_size[1])
            new_w = max(new_w, self.target_size[0])

        img_resized = cv2.resize(img, (new_w, new_h))
        start_x = (new_w - self.target_size[0]) // 2
        start_y = (new_h - self.target_size[1]) // 2
        img_processed = img_resized[start_y:start_y + self.target_size[1], start_x:start_x + self.target_size[0]]

        if self.mask_center:
            center_x, center_y = self.target_size[0] // 2, self.target_size[1] // 2
            half_size = self.mask_size // 2
            img_processed[center_y - half_size:center_y + half_size, center_x - half_size:center_x + half_size] = 0
        
        if output_path:
            cv2.imwrite(output_path, img_processed)
        
        return img_processed
    
    def process_folder(self, input_folder, output_folder):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        files = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg'))
        print(f"Processing {len(files)} images from {input_folder} to {output_folder}")

        count = 0
        for file in files:
            try:
                output_file = output_path / file.name
                self.process_single_image(file, output_file)
                count += 1
            except Exception as e:
                print(f"Failed to process {file.name}: {e}")
        print(f"Processed {count} images successfully.")

if __name__ == "__main__":
    processor = ChestXrayProcessor(target_size=(227, 227), mask_center=True, mask_size=50)
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "Viral Pneumonia" / "images"
    processed_dir = base_dir / "data" / "processed" 

    print("Starting processing of chest X-ray images...")
    processor.process_folder(raw_dir, processed_dir)



