import os
import glob

def debug_dataset_structure():
    current_dir = os.getcwd()
    print(f"🔍 Current directory: {current_dir}")
    print("=" * 60)
    
    # List all files and directories in current directory
    print("📁 Contents of current directory:")
    for item in sorted(os.listdir(current_dir)):
        path = os.path.join(current_dir, item)
        if os.path.isdir(path):
            print(f"   📂 {item}/")
        else:
            print(f"   📄 {item}")
    
    print("\n" + "=" * 60)
    
    # Check for train, eval, test directories
    for split_name in ['train', 'eval', 'test']:
        split_path = os.path.join(current_dir, split_name)
        print(f"\n🔍 Checking {split_name} directory:")
        print(f"   Path: {split_path}")
        
        if os.path.exists(split_path):
            print(f"   ✅ Directory exists")
            
            # List all files in the directory
            files = os.listdir(split_path)
            print(f"   📊 Total files: {len(files)}")
            
            if files:
                print(f"   📋 File types:")
                file_types = {}
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext == '':
                        ext = 'no extension'
                    file_types[ext] = file_types.get(ext, 0) + 1
                
                for ext, count in sorted(file_types.items()):
                    print(f"      {ext}: {count} files")
                
                # Show first few files as examples
                print(f"   📄 Example files:")
                for i, file in enumerate(sorted(files)[:5]):
                    file_path = os.path.join(split_path, file)
                    size = os.path.getsize(file_path)
                    print(f"      {file} ({size} bytes)")
                
                if len(files) > 5:
                    print(f"      ... and {len(files) - 5} more files")
            else:
                print(f"   ⚠️ Directory is empty")
            
            # Specifically check for TFRecord files
            tfrecord_patterns = ['*.tfrecord', '*.tfrecords', '*.tf']
            tfrecord_files = []
            for pattern in tfrecord_patterns:
                tfrecord_files.extend(glob.glob(os.path.join(split_path, pattern)))
            
            print(f"   🎯 TFRecord files: {len(tfrecord_files)}")
            if tfrecord_files:
                for tfr_file in tfrecord_files[:3]:  # Show first 3
                    basename = os.path.basename(tfr_file)
                    size = os.path.getsize(tfr_file)
                    print(f"      ✅ {basename} ({size} bytes)")
                if len(tfrecord_files) > 3:
                    print(f"      ... and {len(tfrecord_files) - 3} more TFRecord files")
            else:
                print(f"      ❌ No TFRecord files found")
        else:
            print(f"   ❌ Directory does not exist")
    
    print("\n" + "=" * 60)
    
    # Check for other possible data directories
    possible_data_dirs = [
        "content",
        "data",
        "dataset",
        "datasets",
        "retina_data",
        "diabetic_retinopathy"
    ]
    
    print("🔍 Checking for other possible data directories:")
    for dir_name in possible_data_dirs:
        dir_path = os.path.join(current_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"   ✅ Found: {dir_name}/")
            # Check if it has train/eval/test subdirs
            for split in ['train', 'eval', 'test']:
                split_path = os.path.join(dir_path, split)
                if os.path.exists(split_path):
                    tfrecords = glob.glob(os.path.join(split_path, "*.tfrecord"))
                    print(f"      📂 {split}/ ({len(tfrecords)} TFRecord files)")
        else:
            print(f"   ❌ Not found: {dir_name}/")
    
    print("\n" + "=" * 60)
    print("🏁 Summary:")
    
    # Final check and recommendation
    train_tfrecords = glob.glob(os.path.join(current_dir, "train", "*.tfrecord"))
    eval_tfrecords = glob.glob(os.path.join(current_dir, "eval", "*.tfrecord"))
    test_tfrecords = glob.glob(os.path.join(current_dir, "test", "*.tfrecord"))
    
    print(f"   Train TFRecords: {len(train_tfrecords)}")
    print(f"   Eval TFRecords: {len(eval_tfrecords)}")
    print(f"   Test TFRecords: {len(test_tfrecords)}")
    
    if train_tfrecords and eval_tfrecords and test_tfrecords:
        print("   ✅ Dataset structure looks good!")
        return True
    else:
        print("   ❌ Missing TFRecord files in one or more splits")
        print("\n💡 Possible solutions:")
        print("   1. Make sure your TFRecord files have .tfrecord extension")
        print("   2. Check if files are in a subdirectory")
        print("   3. Verify the files aren't corrupted")
        print("   4. Consider using TensorFlow Datasets (TFDS) instead")
        return False

if __name__ == "__main__":
    print("🔬 Dataset Structure Debug Tool")
    print("=" * 60)
    debug_dataset_structure()