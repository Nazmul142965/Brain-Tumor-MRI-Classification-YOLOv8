import os
import shutil
import random

def split_brain_mri_data(source_root, output_root, train_split=0.8):
    """
    Automated pipeline to restructure the Masoud Nickparvar dataset 
    for YOLOv8-Nano classification.
    """
    # Define the 4 classes exactly as they appear in the dataset
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Create the directory structure required by YOLOv8
    for folder in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(output_root, folder, cls), exist_ok=True)

    for cls in classes:
        src_path = os.path.join(source_root, cls)
        
        # Gather all image files (.jpg format)
        images = [f for f in os.listdir(src_path) if f.lower().endswith('.jpg')]
        
        # Shuffle for a randomized, unbiased split
        random.shuffle(images)
        
        # Calculate the 80% split point
        split_point = int(len(images) * train_split)
        train_files = images[:split_point]
        val_files = images[split_point:]

        # Move files to the training directory
        for img in train_files:
            shutil.copy(os.path.join(src_path, img), 
                        os.path.join(output_root, 'train', cls, img))
            
        # Move files to the validation directory
        for img in val_files:
            shutil.copy(os.path.join(src_path, img), 
                        os.path.join(output_root, 'val', cls, img))
            
        print(f"✅ {cls.capitalize()}: {len(train_files)} training, {len(val_files)} validation images.")

if __name__ == "__main__":
    # Ensure these paths match your local or Colab environment
    SOURCE = "temp_extracted/Training" 
    OUTPUT = "yolo_dataset"
    
    split_brain_mri_data(SOURCE, OUTPUT)
    print("\nDataset successfully split 80/20 for YOLOv8-Nano training.")
