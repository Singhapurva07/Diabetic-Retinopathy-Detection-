import os
import glob

def explore_directory_recursive(path, max_depth=3, current_depth=0):
    """Recursively explore directory structure"""
    if current_depth > max_depth:
        return
    
    indent = "  " * current_depth
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            if os.path.isdir(item_path):
                print(f"{indent}📂 {item}/")
                # Check for TFRecord files in this directory
                tfrecord_files = glob.glob(os.path.join(item_path, "*.tfrecord"))
                if tfrecord_files:
                    print(f"{indent}   🎯 {len(tfrecord_files)} TFRecord files found!")
                
                # Recurse into subdirectory
                explore_directory_recursive(item_path, max_depth, current_depth + 1)
            else:
                file_size = os.path.getsize(item_path)
                file_ext = os.path.splitext(item)[1].lower()
                
                if file_ext in ['.tfrecord', '.tfrecords', '.tf']:
                    print(f"{indent}🎯 {item} ({file_size} bytes) - TFRecord file!")
                else:
                    print(f"{indent}📄 {item} ({file_size} bytes)")
    except PermissionError:
        print(f"{indent}❌ Permission denied")
    except Exception as e:
        print(f"{indent}❌ Error: {e}")

def find_tfrecords_everywhere():
    """Find all TFRecord files in the entire project directory"""
    current_dir = os.getcwd()
    print("🔍 Searching for TFRecord files everywhere...")
    
    # Search for all TFRecord files
    patterns = ["**/*.tfrecord", "**/*.tfrecords", "**/*.tf"]
    all_tfrecords = []
    
    for pattern in patterns:
        tfrecords = glob.glob(os.path.join(current_dir, pattern), recursive=True)
        all_tfrecords.extend(tfrecords)
    
    if all_tfrecords:
        print(f"🎉 Found {len(all_tfrecords)} TFRecord files:")
        for tfr in sorted(all_tfrecords):
            rel_path = os.path.relpath(tfr, current_dir)
            size = os.path.getsize(tfr)
            print(f"   ✅ {rel_path} ({size} bytes)")
    else:
        print("❌ No TFRecord files found anywhere in the project")
    
    return all_tfrecords

def main():
    current_dir = os.getcwd()
    print("🔬 Deep Content Directory Explorer")
    print("=" * 60)
    print(f"📂 Current directory: {current_dir}")
    
    # First, search for TFRecords everywhere
    all_tfrecords = find_tfrecords_everywhere()
    
    print("\n" + "=" * 60)
    print("📂 Exploring content/ directory structure:")
    
    content_path = os.path.join(current_dir, "content")
    if os.path.exists(content_path):
        explore_directory_recursive(content_path, max_depth=4)
    else:
        print("❌ content/ directory not found")
    
    print("\n" + "=" * 60)
    print("🎯 Dataset Structure Analysis:")
    
    # Analyze the found TFRecord files
    if all_tfrecords:
        # Group by parent directory
        dir_groups = {}
        for tfr in all_tfrecords:
            parent_dir = os.path.dirname(tfr)
            if parent_dir not in dir_groups:
                dir_groups[parent_dir] = []
            dir_groups[parent_dir].append(tfr)
        
        print("📊 TFRecord files grouped by directory:")
        for dir_path, files in dir_groups.items():
            rel_dir = os.path.relpath(dir_path, current_dir)
            print(f"   📂 {rel_dir}/ ({len(files)} files)")
            
            # Check if this looks like train/eval/test structure
            dir_name = os.path.basename(dir_path).lower()
            if dir_name in ['train', 'eval', 'test', 'validation']:
                print(f"      🎯 This looks like a {dir_name.upper()} dataset!")
        
        # Suggest the correct data directory
        print("\n💡 Recommended data directory paths:")
        unique_parent_dirs = set(os.path.dirname(os.path.dirname(tfr)) for tfr in all_tfrecords)
        for parent_dir in unique_parent_dirs:
            rel_parent = os.path.relpath(parent_dir, current_dir)
            
            # Check if this parent has train/eval/test subdirs
            has_train = any('train' in tfr for tfr in all_tfrecords if parent_dir in tfr)
            has_eval = any('eval' in tfr for tfr in all_tfrecords if parent_dir in tfr)
            has_test = any('test' in tfr for tfr in all_tfrecords if parent_dir in tfr)
            
            if has_train and has_eval and has_test:
                print(f"   🎉 PERFECT: {rel_parent}")
                print(f"      This directory contains train/, eval/, and test/ subdirectories with TFRecords!")
            else:
                print(f"   ⚠️  {rel_parent}")
                print(f"      Has: {'train ' if has_train else ''}{'eval ' if has_eval else ''}{'test' if has_test else ''}")
    else:
        print("❌ No TFRecord files found to analyze")
        print("\n💡 Possible solutions:")
        print("   1. Download/create TFRecord files")
        print("   2. Use TensorFlow Datasets (TFDS)")
        print("   3. Convert your images to TFRecord format")
        print("   4. Use a different data format (images + CSV)")

if __name__ == "__main__":
    main()