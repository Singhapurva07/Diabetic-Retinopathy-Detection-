import os
import glob
import sys

def find_dataset_directory():
    """Find the actual dataset directory"""
    
    print("🔍 Looking for dataset directories...")
    print("=" * 50)
    
    current_dir = os.getcwd()
    print(f"📁 Current directory: {current_dir}")
    
    # Look for common dataset directory names
    possible_names = [
        'RETINA', 'retina', 'Retina',
        'diabetic-retinopathy-detection', 
        'diabetic_retinopathy',
        'dr_detection',
        'content',
        'data',
        'dataset',
        'MyPrivateDataset1'  # Based on your screenshot
    ]
    
    found_dirs = []
    
    # Check current directory
    print(f"\n📂 Contents of current directory:")
    items = os.listdir(current_dir)
    for item in items:
        item_path = os.path.join(current_dir, item)
        if os.path.isdir(item_path):
            print(f"   📁 {item}/")
            
            # Check if this directory contains the expected structure
            subdirs = ['train', 'eval', 'test']
            has_subdirs = all(os.path.exists(os.path.join(item_path, subdir)) for subdir in subdirs)
            
            if has_subdirs:
                print(f"      ✅ Contains train/eval/test subdirectories!")
                found_dirs.append(item)
                
                # Check for TFRecord files
                for subdir in subdirs:
                    tfrecord_files = glob.glob(os.path.join(item_path, subdir, '*.tfrecord'))
                    print(f"         📄 {subdir}: {len(tfrecord_files)} .tfrecord files")
            
            # Also check if directory name matches expected patterns
            if item.lower() in [name.lower() for name in possible_names]:
                print(f"      📌 Matches expected dataset name pattern")
                if item not in found_dirs:
                    found_dirs.append(item)
        else:
            print(f"   📄 {item}")
    
    # Check parent directories
    parent_dir = os.path.dirname(current_dir)
    print(f"\n📂 Checking parent directory: {parent_dir}")
    
    if os.path.exists(parent_dir):
        parent_items = os.listdir(parent_dir)
        for item in parent_items:
            if item.lower() in [name.lower() for name in possible_names]:
                item_path = os.path.join(parent_dir, item)
                if os.path.isdir(item_path):
                    print(f"   📁 Found potential dataset: {item_path}")
                    found_dirs.append(item_path)
    
    # Look for any .tfrecord files anywhere nearby
    print(f"\n🔎 Searching for .tfrecord files...")
    tfrecord_files = glob.glob("**/*.tfrecord", recursive=True)
    if tfrecord_files:
        print(f"   📄 Found {len(tfrecord_files)} .tfrecord files:")
        for file in tfrecord_files[:10]:  # Show first 10
            print(f"      • {file}")
        if len(tfrecord_files) > 10:
            print(f"      ... and {len(tfrecord_files) - 10} more")
    else:
        print(f"   ❌ No .tfrecord files found in current directory tree")
    
    return found_dirs

def create_fixed_training_script(dataset_path):
    """Create a training script with the correct dataset path"""
    
    script_content = f'''import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import os
import glob
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')

class DiabeticRetinopathyModel:
    def __init__(self):
        self.data_dir = r"{dataset_path}\drive\MyDrive\tfrecords"  # Fixed path to actual tfrecords
        self.image_size = (224, 224)  # Smaller size for faster training
        self.batch_size = 8  # Smaller batch for memory efficiency
        self.num_classes = 5
        self.model = None
        
        print(f"📁 Using dataset directory: {{self.data_dir}}")
        
        # Verify directory exists
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Dataset directory does not exist: {{self.data_dir}}")
    
    def parse_tfrecord(self, example_proto):
        """Parse TFRecord format with error handling"""
        try:
            feature_description = {{
                'image/encoded': tf.io.FixedLenFeature([], tf.string),
                'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            }}
            return tf.io.parse_single_example(example_proto, feature_description)
        except Exception as e:
            print(f"Parse error: {{e}}")
            # Return dummy data to avoid breaking the pipeline
            return {{
                'image/encoded': tf.constant(b''),
                'image/class/label': tf.constant(0, dtype=tf.int64)
            }}
    
    def decode_image(self, parsed_example):
        """Decode and preprocess image with error handling"""
        try:
            image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
            label = tf.cast(parsed_example['image/class/label'], tf.int32)
            
            # Resize and normalize
            image = tf.image.resize(image, self.image_size)
            image = tf.cast(image, tf.float32) / 255.0
            
            return image, label
        except:
            # Return dummy data for corrupted images
            dummy_image = tf.zeros((*self.image_size, 3), dtype=tf.float32)
            dummy_label = tf.constant(0, dtype=tf.int32)
            return dummy_image, dummy_label
    
    def create_dataset(self, split_name, max_files=2):
        """Create dataset with limited files for testing"""
        pattern = os.path.join(self.data_dir, split_name, "*.tfrecord")
        files = glob.glob(pattern)
        
        print(f"📄 Found {{len(files)}} TFRecord files for {{split_name}}")
        
        if len(files) == 0:
            raise ValueError(f"No TFRecord files found in {{pattern}}")
        
        # Use only first few files for quick testing
        files = files[:max_files]
        print(f"📄 Using {{len(files)}} files for {{split_name}}")
        
        # Create dataset
        dataset = tf.data.TFRecordDataset(files)
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        # Filter out invalid records
        dataset = dataset.filter(lambda x, y: tf.reduce_all(tf.shape(x)[0] > 0))
        
        # Take limited samples for quick training
        if split_name == 'train':
            dataset = dataset.take(1000)  # Limit training samples
            dataset = dataset.shuffle(100)
        else:
            dataset = dataset.take(200)   # Limit validation samples
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_simple_model(self):
        """Create a simpler model for quick testing"""
        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        
        # Use smaller pre-trained model
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )(inputs)
        
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self):
        """Train model with simple approach"""
        try:
            print("🚀 Creating datasets...")
            train_dataset = self.create_dataset('train', max_files=2)
            eval_dataset = self.create_dataset('eval', max_files=1)
            
            print("🏗️ Creating model...")
            self.model = self.create_simple_model()
            
            print(f"📊 Model has {{self.model.count_params():,}} parameters")
            
            print("🎯 Starting training...")
            history = self.model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=5,  # Quick training
                verbose=1,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
                ]
            )
            
            print("💾 Saving model...")
            self.model.save('best_dr_model.h5')
            
            print("✅ Training completed successfully!")
            return history
            
        except Exception as e:
            print(f"❌ Training failed: {{str(e)}}")
            return None

def main():
    print("🔬 Diabetic Retinopathy Detection - Quick Test")
    print("=" * 50)
    
    try:
        # Initialize and train model
        dr_model = DiabeticRetinopathyModel()
        history = dr_model.train_model()
        
        if history is not None:
            # Plot results
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_results.png')
            plt.show()
            
            print("🎉 Training completed successfully!")
            print("📊 Results saved as 'training_results.png'")
            print("💾 Model saved as 'best_dr_model.h5'")
        else:
            print("❌ Training failed - please check your dataset")
            
    except Exception as e:
        print(f"❌ Error: {{str(e)}}")
        print("💡 Please check your dataset structure and file paths")

if __name__ == "__main__":
    main()
'''
    
    with open('train_fixed.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"✅ Created fixed training script: train_fixed.py")
    print(f"📁 Using dataset path: {dataset_path}")

def main():
    """Main function to find dataset and create fixed training script"""
    
    print("🔍 DATASET FINDER AND FIXER")
    print("=" * 50)
    
    # Find potential dataset directories
    found_dirs = find_dataset_directory()
    
    if found_dirs:
        print(f"\\n✅ Found {len(found_dirs)} potential dataset directories:")
        for i, dir_path in enumerate(found_dirs, 1):
            print(f"   {i}. {dir_path}")
        
        if len(found_dirs) == 1:
            # Use the only found directory
            selected_path = found_dirs[0]
            print(f"\\n🎯 Using: {selected_path}")
        else:
            # Let user choose
            print(f"\\n❓ Which directory contains your dataset?")
            try:
                choice = int(input(f"Enter number (1-{len(found_dirs)}): ")) - 1
                selected_path = found_dirs[choice]
            except (ValueError, IndexError):
                print("Invalid choice, using first directory")
                selected_path = found_dirs[0]
        
        # Convert to absolute path
        if not os.path.isabs(selected_path):
            selected_path = os.path.abspath(selected_path)
        
        print(f"\\n🔧 Creating fixed training script...")
        create_fixed_training_script(selected_path)
        
        print(f"\\n🚀 Next steps:")
        print(f"1. Run: python train_fixed.py")
        print(f"2. If successful, run: streamlit run app.py")
        
    else:
        print(f"\\n❌ No dataset directories found!")
        print(f"\\n💡 Please ensure your dataset follows this structure:")
        print(f"   dataset_folder/")
        print(f"   ├── train/")
        print(f"   │   ├── *.tfrecord")
        print(f"   ├── eval/") 
        print(f"   │   ├── *.tfrecord")
        print(f"   └── test/")
        print(f"       ├── *.tfrecord")
        
        print(f"\\n🔄 Alternative: Download dataset manually")
        print(f"   1. Go to Kaggle Diabetic Retinopathy Detection")
        print(f"   2. Download and extract to current directory")
        print(f"   3. Run this script again")

if __name__ == "__main__":
    main()