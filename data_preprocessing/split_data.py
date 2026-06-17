import os
import shutil
import random

def split_brain_mri_data(source_root, output_root, train_split=0.8):
    """
    Optimized case-insensitive pipeline to restructure the Masoud Nickparvar dataset 
    for YOLOv8-Nano classification.
    """
    classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # 1. Create the target directory structure required by YOLOv8
    for folder in ['train', 'val']:
        for cls in classes:
            os.makedirs(os.path.join(output_root, folder, cls), exist_ok=True)

    # 2. Process each class dynamically
    for cls in classes:
        # Match class names dynamically regardless of uppercase/lowercase in source
        try:
            actual_folder_name = next(d for d in os.listdir(source_root) if d.lower() == cls)
            src_path = os.path.join(source_root, actual_folder_name)
        except StopIteration:
            print(f"❌ Error: Could not find a folder matching '{cls}' in {source_root}")
            continue
        
        # Gather all image variants (.jpg, .png, .jpeg)
        images = [f for f in os.listdir(src_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        random.shuffle(images)
        
        # Calculate the 80/20 split points
        split_point = int(len(images) * train_split)
        train_files = images[:split_point]
        val_files = images[split_point:]

        # Copy files to respective directories
        for img in train_files:
            shutil.copy(os.path.join(src_path, img), os.path.join(output_root, 'train', cls, img))
            
        for img in val_files:
            shutil.copy(os.path.join(src_path, img), os.path.join(output_root, 'val', cls, img))
            
        print(f"✅ {cls.capitalize()}: {len(train_files)} training, {len(val_files)} validation images.")

if __name__ == "__main__":
    SOURCE = "temp_extracted/Training" 
    OUTPUT = "yolo_dataset"
    
    # Wipe old folder if it exists to clean setup
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
        
    split_brain_mri_data(SOURCE, OUTPUT)
    print("\nDataset successfully split 80/20 for YOLOv8-Nano training.")
