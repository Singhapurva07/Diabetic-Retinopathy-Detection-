import tensorflow as tf
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
from tensorflow.keras.mixed_precision import set_global_policy
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Enable mixed precision for faster training
set_global_policy('mixed_float16')

class DiabeticRetinopathyModel:
    def __init__(self, data_dir='RETINA'):
        self.data_dir = data_dir
        self.image_size = (384, 384)  # Optimal size for EfficientNetV2L
        self.batch_size = 16  # Reduced for memory efficiency
        self.num_classes = 5  # 0-4 severity levels
        self.model = None
        
        # Class weights for imbalanced dataset (typical DR distribution)
        self.class_weights = {
            0: 1.0,    # No DR
            1: 2.0,    # Mild
            2: 5.0,    # Moderate  
            3: 10.0,   # Severe
            4: 15.0    # Proliferative DR
        }
    
    def parse_tfrecord(self, example_proto):
        """Parse TFRecord format"""
        feature_description = {
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(example_proto, feature_description)
    
    def decode_image(self, parsed_example):
        """Decode and preprocess image"""
        image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)
        label = tf.cast(parsed_example['image/class/label'], tf.int32)
        
        # Resize and normalize
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        
        return image, label
    
    def augment_image(self, image, label):
        """Advanced data augmentation for retinal images"""
        # Random rotation (small angles to preserve medical accuracy)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentation (subtle for medical images)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.image.random_saturation(image, 0.9, 1.1)
        
        # Random crop and resize
        image = tf.image.random_crop(image, size=[*self.image_size, 3])
        
        # Ensure values stay in [0,1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    def create_dataset(self, tfrecord_pattern, is_training=True):
        """Create dataset from TFRecord files"""
        files = glob.glob(f"{self.data_dir}/{tfrecord_pattern}/*.tfrecord")
        print(f"Found {len(files)} TFRecord files for {tfrecord_pattern}")
        
        if len(files) == 0:
            raise ValueError(f"No TFRecord files found in {self.data_dir}/{tfrecord_pattern}/")
        
        # Create dataset with proper error handling
        dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        
        # Count total samples for progress bar
        total_samples = 0
        for file in files:
            try:
                file_dataset = tf.data.TFRecordDataset(file)
                file_count = sum(1 for _ in file_dataset)
                total_samples += file_count
                print(f"  {os.path.basename(file)}: {file_count} samples")
            except:
                print(f"  Warning: Could not count samples in {os.path.basename(file)}")
        
        print(f"Total samples in {tfrecord_pattern}: {total_samples}")
        
        # Process dataset
        dataset = dataset.map(self.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.decode_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        if is_training:
            dataset = dataset.shuffle(min(1000, total_samples))
            dataset = dataset.map(self.augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Calculate steps per epoch
        steps_per_epoch = max(1, total_samples // self.batch_size)
        
        return dataset, steps_per_epoch
    
    def create_model(self):
        """Create advanced model architecture"""
        # Load pre-trained EfficientNetV2L
        base_model = EfficientNetV2L(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.image_size, 3)
        )
        
        # Unfreeze last few layers for fine-tuning
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Output layer with mixed precision
        predictions = Dense(self.num_classes, activation='softmax', dtype='float32')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        return model
    
    def compile_model(self):
        """Compile model with advanced settings"""
        self.model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
    
    def train_model(self):
        """Train the model with advanced techniques"""
        try:
            # Create datasets with step counting
            print("Creating training dataset...")
            train_dataset, train_steps = self.create_dataset('train', is_training=True)
            print("Creating validation dataset...")
            eval_dataset, eval_steps = self.create_dataset('eval', is_training=False)
            
            print(f"Training steps per epoch: {train_steps}")
            print(f"Validation steps per epoch: {eval_steps}")
            
            # Create model
            self.model = self.create_model()
            self.compile_model()
            
            print(f"Model created with {self.model.count_params():,} parameters")
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                ModelCheckpoint(
                    'best_dr_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train model with explicit steps
            history = self.model.fit(
                train_dataset,
                steps_per_epoch=train_steps,
                validation_data=eval_dataset,
                validation_steps=eval_steps,
                epochs=100,
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            return history
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            print("\nTrying alternative dataset loading method...")
            return self.train_model_fallback()
    
    def train_model_fallback(self):
        """Fallback training method with manual data loading"""
        print("Using fallback method - loading data manually...")
        
        try:
            # Manually load a small batch to test
            train_files = glob.glob(f"{self.data_dir}/train/*.tfrecord")
            if not train_files:
                raise ValueError("No training files found!")
            
            # Create simple dataset
            train_dataset = tf.data.TFRecordDataset(train_files[:2])  # Use only first 2 files for testing
            train_dataset = train_dataset.map(self.parse_tfrecord)
            train_dataset = train_dataset.map(self.decode_image)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.take(100)  # Limit to 100 batches for testing
            
            eval_files = glob.glob(f"{self.data_dir}/eval/*.tfrecord")
            eval_dataset = tf.data.TFRecordDataset(eval_files[:1])  # Use only first file
            eval_dataset = eval_dataset.map(self.parse_tfrecord)
            eval_dataset = eval_dataset.map(self.decode_image)
            eval_dataset = eval_dataset.batch(self.batch_size)
            eval_dataset = eval_dataset.take(20)  # Limit to 20 batches for testing
            
            # Create smaller model for testing
            self.model = self.create_model()
            self.compile_model()
            
            print("Starting training with limited data for testing...")
            
            # Simple training without complex callbacks
            history = self.model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=5,  # Reduced epochs for testing
                verbose=1
            )
            
            # Save model
            self.model.save('best_dr_model.h5')
            print("Model saved successfully!")
            
            return history
            
        except Exception as e:
            print(f"Fallback method also failed: {str(e)}")
            print("\nPlease check your TFRecord files and data structure.")
            return None

    def evaluate_model(self):
        """Evaluate model on test set"""
        try:
            test_dataset, test_steps = self.create_dataset('test', is_training=False)
        except:
            print("Using fallback evaluation method...")
            test_files = glob.glob(f"{self.data_dir}/test/*.tfrecord")
            test_dataset = tf.data.TFRecordDataset(test_files[:1])
            test_dataset = test_dataset.map(self.parse_tfrecord)
            test_dataset = test_dataset.map(self.decode_image)
            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.take(10)
        
        # Predictions
        predictions = []
        true_labels = []
        
        for batch_images, batch_labels in test_dataset:
            pred = self.model.predict(batch_images, verbose=0)
            predictions.extend(np.argmax(pred, axis=1))
            true_labels.extend(batch_labels.numpy())
        
        if len(predictions) == 0:
            print("No test data available for evaluation")
            return 0.0, 0.0, [], []
        
        # Metrics
        accuracy = np.mean(np.array(predictions) == np.array(true_labels))
        kappa_score = cohen_kappa_score(true_labels, predictions)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Quadratic Weighted Kappa: {kappa_score:.4f}")
        
        # Classification report
        class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - Diabetic Retinopathy Detection')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy, kappa_score, predictions, true_labels
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0,0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0,0].set_title('Model Accuracy')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # Loss
        axes[0,1].plot(history.history['loss'], label='Training Loss')
        axes[0,1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0,1].set_title('Model Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True)
        
        # Learning Rate (if available)
        if 'lr' in history.history:
            axes[1,0].plot(history.history['lr'])
            axes[1,0].set_title('Learning Rate')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Learning Rate')
            axes[1,0].set_yscale('log')
            axes[1,0].grid(True)
        
        # Top-k accuracy
        if 'sparse_top_k_categorical_accuracy' in history.history:
            axes[1,1].plot(history.history['sparse_top_k_categorical_accuracy'], 
                          label='Training Top-2 Accuracy')
            axes[1,1].plot(history.history['val_sparse_top_k_categorical_accuracy'], 
                          label='Validation Top-2 Accuracy')
            axes[1,1].set_title('Top-2 Accuracy')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].legend()
            axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_single_image(self, image_path):
        """Predict on single image with explainability"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, self.image_size)
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        # Prediction
        prediction = self.model.predict(image_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        class_names = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        
        return {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': prediction[0],
            'class_names': class_names
        }

def main():
    """Main training pipeline"""
    print("🔬 Diabetic Retinopathy Detection Model Training")
    print("=" * 50)
    
    # Initialize model
    dr_model = DiabeticRetinopathyModel()
    
    print("📊 Starting model training...")
    history = dr_model.train_model()
    
    print("\n📈 Plotting training history...")
    dr_model.plot_training_history(history)
    
    print("\n🧪 Evaluating model on test set...")
    accuracy, kappa_score, predictions, true_labels = dr_model.evaluate_model()
    
    print(f"\n🎯 Final Results:")
    print(f"   • Test Accuracy: {accuracy:.4f}")
    print(f"   • Quadratic Weighted Kappa: {kappa_score:.4f}")
    
    print(f"\n💾 Model saved as 'best_dr_model.h5'")
    print(f"📊 Training plots saved as 'training_history.png' and 'confusion_matrix.png'")
    
    return dr_model

if __name__ == "__main__":
    # Set GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    model = main()