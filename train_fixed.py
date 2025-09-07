import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from collections import Counter
import os
import glob
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, LayerNormalization, MultiHeadAttention, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import albumentations as A
import cv2
import warnings
warnings.filterwarnings('ignore')

class AdvancedDiabeticRetinopathyModel:
    def __init__(self, use_tfds=False, model_type='efficientnetv2l'):
        self.use_tfds = use_tfds
        
        # Set the correct data directory path
        current_dir = os.getcwd()
        self.data_dir = os.path.join(current_dir, "content", "drive", "MyDrive", "tfrecords")
        
        self.image_size = (512, 512)
        self.batch_size = 8
        self.num_classes = 5
        self.model = None
        self.model_type = model_type.lower()
        
        self.augmentation = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.75),
            A.Blur(blur_limit=3, p=0.3),
            A.OpticalDistortion(p=0.3),
            A.GridDistortion(p=0.3),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        ])
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        if not self.use_tfds:
            print(f"Using dataset directory: {self.data_dir}")
            self._verify_dataset_structure()
        else:
            try:
                import tensorflow_datasets as tfds
                print("Using TensorFlow Datasets: diabetic_retinopathy_detection")
            except ImportError:
                print("tensorflow-datasets not installed! Falling back to local TFRecords")
                self.use_tfds = False
                self._verify_dataset_structure()

    def _verify_dataset_structure(self):
        """Verify and display dataset structure information"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory {self.data_dir} not found")
        
        total_files = 0
        for split in ['train', 'eval', 'test']:
            split_path = os.path.join(self.data_dir, split)
            if os.path.exists(split_path):
                files = glob.glob(os.path.join(split_path, "*.tfrecord"))
                print(f"   {split}: {len(files)} TFRecord files")
                total_files += len(files)
            else:
                print(f"   WARNING: {split} directory missing")
        
        if total_files == 0:
            raise FileNotFoundError("No TFRecord files found in any split directory")
        
        # Inspect the first available TFRecord file
        files = glob.glob(os.path.join(self.data_dir, 'train', '*.tfrecord'))
        if files:
            print("Inspecting TFRecord contents...")
            self.inspect_tfrecords(files)

    def inspect_tfrecords(self, files):
        dataset = tf.data.TFRecordDataset(files[0])
        valid_samples = 0
        invalid_samples = 0
        for raw_record in dataset.take(100):
            try:
                example = tf.train.Example()
                example.ParseFromString(raw_record.numpy())
                parsed = self.parse_tfrecord(raw_record)
                if parsed is None:
                    invalid_samples += 1
                    continue
                image, label = self.decode_image(parsed)
                if tf.reduce_any(tf.equal(tf.shape(image), 0)) or label < 0 or label >= self.num_classes:
                    invalid_samples += 1
                else:
                    valid_samples += 1
            except Exception as e:
                invalid_samples += 1
        print(f"TFRecord inspection: {valid_samples} valid, {invalid_samples} invalid samples (out of 100)")

    def parse_tfrecord(self, example_proto):
        """Parse TFRecord with multiple possible schemas"""
        feature_descriptions = [
            {'image': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)},
            {'image/encoded': tf.io.FixedLenFeature([], tf.string), 'image/class/label': tf.io.FixedLenFeature([], tf.int64)}
        ]
        for feature_description in feature_descriptions:
            try:
                parsed = tf.io.parse_single_example(example_proto, feature_description)
                if 'image/encoded' in parsed:
                    parsed['image'] = parsed['image/encoded']
                    parsed['label'] = parsed['image/class/label']
                return parsed
            except Exception as e:
                continue
        # Return empty tensors with correct types instead of None
        return {
            'image': tf.constant(b'', dtype=tf.string),
            'label': tf.constant(-1, dtype=tf.int64)
        }
    
    def advanced_image_preprocessing(self, image_tensor):
        """Advanced image preprocessing with proper error handling"""
        def _preprocess_py(image_bytes):
            try:
                # Decode image
                image_np = tf.io.decode_image(image_bytes, channels=3, dtype=tf.uint8).numpy()
                if len(image_np.shape) != 3 or image_np.shape[2] != 3:
                    # Return black image with correct shape
                    return np.zeros((*self.image_size, 3), dtype=np.uint8)
                
                # Apply CLAHE
                lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
                lab[:,:,0] = self.clahe.apply(lab[:,:,0])
                image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                # Find retina mask and crop
                mask = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                mask = mask > (mask.mean() / 3)
                coords = cv2.findNonZero(mask.astype(np.uint8))
                if coords is not None:
                    x, y, w, h = cv2.boundingRect(coords)
                    if w > 0 and h > 0:
                        image_np = image_np[y:y+h, x:x+w]
                
                return image_np.astype(np.uint8)
            except Exception as e:
                # Return black image with correct shape
                return np.zeros((*self.image_size, 3), dtype=np.uint8)
        
        processed_image = tf.py_function(
            func=_preprocess_py, 
            inp=[image_tensor], 
            Tout=tf.uint8
        )
        processed_image.set_shape([None, None, 3])
        return processed_image
    
    def decode_image(self, parsed_example):
        """Decode image with guaranteed output types"""
        # Check if this is a valid sample
        is_valid = tf.logical_and(
            tf.not_equal(parsed_example['image'], b''),
            tf.logical_and(
                parsed_example['label'] >= 0,
                parsed_example['label'] < self.num_classes
            )
        )
        
        def process_valid():
            image = self.advanced_image_preprocessing(parsed_example['image'])
            image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1])
            image = tf.cast(image, tf.float32) / 255.0
            image = tf.image.per_image_standardization(image)
            label = tf.cast(parsed_example['label'], tf.int32)
            return image, label
        
        def process_invalid():
            # Return dummy data with correct types and shapes
            image = tf.zeros((*self.image_size, 3), dtype=tf.float32)
            label = tf.constant(-1, dtype=tf.int32)  # Invalid label for filtering
            return image, label
        
        return tf.cond(is_valid, process_valid, process_invalid)
    
    def filter_valid_samples(self, image, label):
        """Filter out invalid samples"""
        return label >= 0
    
    def advanced_augmentation(self, image, label):
        """Apply advanced augmentation"""
        def augment_fn(image_tensor):
            try:
                image_np = (image_tensor.numpy() * 255).astype(np.uint8)
                augmented = self.augmentation(image=image_np)
                return augmented['image'].astype(np.float32) / 255.0
            except Exception as e:
                return image_tensor.numpy()
        
        augmented_image = tf.py_function(func=augment_fn, inp=[image], Tout=tf.float32)
        augmented_image.set_shape(self.image_size + (3,))
        return augmented_image, label
    
    def mixup(self, batch_x, batch_y, alpha=0.2):
        """Apply mixup augmentation"""
        try:
            batch_size = tf.shape(batch_x)[0]
            lambda_val = tf.random.gamma([batch_size, 1, 1, 1], alpha=alpha)
            lambda_val = tf.maximum(lambda_val, 1 - lambda_val)
            indices = tf.random.shuffle(tf.range(batch_size))
            mixed_x = lambda_val * batch_x + (1 - lambda_val) * tf.gather(batch_x, indices)
            batch_y_onehot = tf.one_hot(batch_y, self.num_classes)
            mixed_y = lambda_val[:, 0, 0, 0:1] * batch_y_onehot + (1 - lambda_val[:, 0, 0, 0:1]) * tf.gather(batch_y_onehot, indices)
            return mixed_x, mixed_y
        except Exception as e:
            return batch_x, tf.one_hot(batch_y, self.num_classes)
    
    def create_dataset(self, split_name, take_samples=None):
        """Create dataset with improved error handling"""
        try:
            pattern = os.path.join(self.data_dir, split_name, "*.tfrecord")
            files = glob.glob(pattern)
            print(f"Loading {split_name} dataset: {len(files)} TFRecord files")
            if len(files) == 0:
                raise ValueError(f"No TFRecord files found in {pattern}")
            
            # Create dataset from TFRecord files
            dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
            
            # Parse and decode
            dataset = dataset.map(
                self.parse_tfrecord, 
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )
            dataset = dataset.map(
                self.decode_image, 
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=False
            )
            
            # Filter out invalid samples
            dataset = dataset.filter(self.filter_valid_samples)
            
            if take_samples:
                dataset = dataset.take(take_samples)
                print(f"   Limited to {take_samples} samples")
            
            # Shuffle and augment for training
            if split_name == 'train':
                dataset = dataset.shuffle(8000, reshuffle_each_iteration=True)
                dataset = dataset.map(
                    self.advanced_augmentation, 
                    num_parallel_calls=tf.data.AUTOTUNE,
                    deterministic=False
                )
                dataset = dataset.repeat()
            
            # Batch the data
            dataset = dataset.batch(self.batch_size, drop_remainder=True)
            
            # Apply mixup for training
            if split_name == 'train':
                dataset = dataset.map(
                    lambda x, y: self.mixup(x, y), 
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            # Try to estimate dataset size
            try:
                if split_name != 'train':  # Don't count infinite train dataset
                    sample_count = 0
                    for _ in dataset.unbatch().take(1000):
                        sample_count += 1
                    print(f"   {split_name} dataset size: {sample_count} samples")
            except Exception as e:
                print(f"   Warning: Could not count samples: {e}")
            
            return dataset
        except Exception as e:
            print(f"Dataset creation failed for {split_name}: {e}")
            raise
    
    def compute_class_weights(self, dataset):
        """Compute class weights for balanced training"""
        labels = []
        try:
            print("Computing class weights...")
            sample_count = 0
            for _, label in dataset.unbatch().take(2000):
                labels.append(label.numpy())
                sample_count += 1
                if sample_count % 500 == 0:
                    print(f"   Processed {sample_count} samples...")
        except Exception as e:
            print(f"Error collecting labels for class weights: {e}")
        
        if not labels:
            print("No valid labels found, using equal weights")
            return {i: 1.0 for i in range(self.num_classes)}
        
        valid_labels = [l for l in labels if 0 <= l < self.num_classes]
        if not valid_labels:
            print("No valid labels in range 0-4, using equal weights")
            return {i: 1.0 for i in range(self.num_classes)}
        
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight('balanced', classes=np.arange(self.num_classes), y=valid_labels)
        class_weights = {i: w for i, w in enumerate(weights)}
        print(f"Class distribution: {dict(sorted(Counter(valid_labels).items()))}")
        print(f"Class weights: {class_weights}")
        return class_weights
    
    def create_attention_block(self, inputs, dim):
        """Create transformer attention block"""
        x = LayerNormalization(epsilon=1e-6)(inputs)
        attention = MultiHeadAttention(num_heads=8, key_dim=dim // 8, dropout=0.1)(x, x)
        x = Add()([attention, inputs])
        ffn_input = LayerNormalization(epsilon=1e-6)(x)
        ffn = Dense(dim * 2, activation='gelu')(ffn_input)
        ffn = Dropout(0.1)(ffn)
        ffn = Dense(dim)(ffn)
        outputs = Add()([ffn, x])
        return outputs
    
    def create_advanced_model(self):
        """Create advanced model with attention mechanism"""
        inputs = tf.keras.Input(shape=(*self.image_size, 3))
        
        # Base model
        if self.model_type == 'efficientnetv2l':
            base_model = EfficientNetV2L(weights='imagenet', include_top=False, input_shape=(*self.image_size, 3))
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        
        base_model.trainable = False
        x = base_model(inputs, training=False)
        
        # Get feature dimensions dynamically
        feature_dim = x.shape[-1]
        batch_size = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]
        
        # Reshape for attention
        x = tf.reshape(x, (batch_size, h * w, feature_dim))
        
        # Add attention blocks
        x = self.create_attention_block(x, feature_dim)
        x = self.create_attention_block(x, feature_dim)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        
        # Classification head
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(512, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax', name='predictions')(x)
        
        model = Model(inputs, outputs)
        return model, base_model
    
    def create_custom_loss(self):
        """Create custom loss function combining focal loss and label smoothing"""
        def combined_loss(y_true, y_pred):
            # Categorical crossentropy
            cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            
            # Focal loss
            alpha = 0.25
            gamma = 2.0
            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            focal_weight = alpha * tf.pow(1 - pt, gamma)
            focal_loss = focal_weight * cce
            
            # Label smoothing
            epsilon = 0.1
            y_true_smooth = y_true * (1 - epsilon) + epsilon / self.num_classes
            smooth_loss = tf.keras.losses.categorical_crossentropy(y_true_smooth, y_pred)
            
            return 0.7 * focal_loss + 0.3 * smooth_loss
        return combined_loss
    
    def get_cosine_schedule(self, initial_lr=1e-4, warmup_epochs=5, total_epochs=100):
        """Create cosine learning rate schedule with warmup"""
        def schedule(epoch):
            if epoch < warmup_epochs:
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                return initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        return schedule
    
    def train_advanced_model(self, quick_test=False):
        """Train the advanced model with two-stage training"""
        try:
            print("Creating advanced datasets...")
            if quick_test:
                train_dataset = self.create_dataset('train', take_samples=800)
                eval_dataset = self.create_dataset('eval', take_samples=200)
                epochs_stage1 = 10
                epochs_stage2 = 15
                steps_per_epoch = 50
                validation_steps = 25
                print("Running quick test with limited data")
            else:
                train_dataset = self.create_dataset('train')
                eval_dataset = self.create_dataset('eval')
                epochs_stage1 = 30
                epochs_stage2 = 50
                steps_per_epoch = 500
                validation_steps = 100
                print("Running full advanced training")
            
            class_weights = self.compute_class_weights(train_dataset)
            
            print("Creating advanced model...")
            self.model, base_model = self.create_advanced_model()
            
            print("Stage 1: Training classification head...")
            custom_loss = self.create_custom_loss()
            optimizer = AdamW(learning_rate=1e-3, weight_decay=0.01)
            self.model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
            print(f"Model has {self.model.count_params():,} parameters")
            
            callbacks_stage1 = [
                EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1, min_delta=0.001),
                ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-7, verbose=1),
                ModelCheckpoint('best_dr_model_head.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
                tf.keras.callbacks.LearningRateScheduler(self.get_cosine_schedule(1e-3, 3, epochs_stage1))
            ]
            
            history1 = self.model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=epochs_stage1,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks_stage1,
                verbose=1
            )
            
            print("Stage 2: Fine-tuning entire model...")
            base_model.trainable = True
            for layer in base_model.layers[:-30]:  # Unfreeze fewer layers
                layer.trainable = False
            
            optimizer = AdamW(learning_rate=1e-5, weight_decay=0.01)
            self.model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
            
            callbacks_stage2 = [
                EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1, min_delta=0.0005),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-8, verbose=1),
                ModelCheckpoint('best_dr_model_advanced.h5', monitor='val_accuracy', save_best_only=True, verbose=1),
                tf.keras.callbacks.LearningRateScheduler(self.get_cosine_schedule(1e-5, 2, epochs_stage2))
            ]
            
            history2 = self.model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=epochs_stage2,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                callbacks=callbacks_stage2,
                verbose=1
            )
            
            # Combine histories
            history = {
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                'loss': history1.history['loss'] + history2.history['loss'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
            }
            
            print("Advanced training completed successfully!")
            return history
        except Exception as e:
            print(f"Training failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_advanced_model(self):
        """Evaluate the trained model"""
        print("Advanced model evaluation...")
        try:
            test_dataset = self.create_dataset('test', take_samples=500)
            predictions = []
            true_labels = []
            prediction_probs = []
            
            print("Generating predictions...")
            for batch_images, batch_labels in test_dataset:
                pred_probs = self.model.predict(batch_images, verbose=0)
                pred_labels = np.argmax(pred_probs, axis=1)
                predictions.extend(pred_labels)
                true_labels.extend(batch_labels.numpy())
                prediction_probs.extend(pred_probs)
            
            if len(predictions) > 0:
                accuracy = np.mean(np.array(predictions) == np.array(true_labels))
                kappa = cohen_kappa_score(true_labels, predictions)
                print(f"Test Accuracy: {accuracy:.4f}")
                print(f"Cohen's Kappa: {kappa:.4f}")
                
                class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
                print("Detailed Classification Report:")
                print(classification_report(true_labels, predictions, target_names=class_names, zero_division=0))
                
                self.plot_advanced_results(true_labels, predictions, prediction_probs, class_names)
                return accuracy, kappa
            else:
                print("No test predictions available")
                return 0.0, 0.0
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return 0.0, 0.0
    
    def plot_advanced_results(self, true_labels, predictions, prediction_probs, class_names):
        """Plot comprehensive evaluation results"""
        plt.figure(figsize=(15, 10))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Normalized Confusion Matrix
        plt.subplot(2, 3, 2)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Normalized Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Class-wise Accuracy
        plt.subplot(2, 3, 3)
        class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
        bars = plt.bar(class_names, class_accuracy, color='skyblue')
        plt.title('Class-wise Accuracy')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        for bar, acc in zip(bars, class_accuracy):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
        
        # Confidence Distribution
        plt.subplot(2, 3, 4)
        confidence_scores = np.max(prediction_probs, axis=1)
        plt.hist(confidence_scores, bins=20, alpha=0.7, color='green')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Max Probability')
        plt.ylabel('Frequency')
        
        # ROC Curves
        plt.subplot(2, 3, 5)
        y_test_bin = label_binarize(true_labels, classes=list(range(len(class_names))))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], np.array(prediction_probs)[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right", fontsize=8)
        
        # Class Distribution
        plt.subplot(2, 3, 6)
        unique, counts = np.unique(true_labels, return_counts=True)
        plt.pie(counts, labels=[class_names[i] for i in unique], autopct='%1.1f%%')
        plt.title('Test Set Class Distribution')
        
        plt.tight_layout()
        plt.savefig('advanced_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history):
        """Plot training history"""
        if history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(15, 5))
        
        # Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(history['accuracy'], label='Training Accuracy', marker='o', linewidth=2, color='blue')
        plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='s', linewidth=2, color='orange')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(1, 3, 2)
        plt.plot(history['loss'], label='Training Loss', marker='o', linewidth=2, color='blue')
        plt.plot(history['val_loss'], label='Validation Loss', marker='s', linewidth=2, color='orange')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Training Summary
        plt.subplot(1, 3, 3)
        metrics = {
            'Final Val Accuracy': f"{history['val_accuracy'][-1]:.4f}",
            'Best Val Accuracy': f"{max(history['val_accuracy']):.4f}",
            'Final Val Loss': f"{history['val_loss'][-1]:.4f}",
            'Best Val Loss': f"{min(history['val_loss']):.4f}",
            'Total Epochs': f"{len(history['accuracy'])}",
        }
        text = '\n'.join([f"{k}: {v}" for k, v in metrics.items()])
        plt.text(0.5, 0.5, text, transform=plt.gca().transAxes, fontsize=12, ha='center', va='center')
        plt.title('Training Summary')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Advanced Diabetic Retinopathy Detection - Training")
    print("=" * 50)
    try:
        dr_model = AdvancedDiabeticRetinopathyModel(use_tfds=False, model_type='efficientnetv2l')
        
        print("\nStarting advanced training...")
        history = dr_model.train_advanced_model(quick_test=True)
        
        if history is not None:
            print("\nPlotting training history...")
            dr_model.plot_training_history(history)
            
            print("\nEvaluating model...")
            test_accuracy, test_kappa = dr_model.evaluate_advanced_model()
            
            print("\nTRAINING COMPLETED!")
            print("=" * 30)
            print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
            print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Cohen's Kappa: {test_kappa:.4f}")
            print(f"Model saved as: best_dr_model_advanced.h5")
            print(f"Training plot saved as: training_history.png")
            print(f"Evaluation plot saved as: advanced_evaluation_results.png")
            print(f"\nNext Steps:")
            print(f"1. Check training_history.png for training plots")
            print(f"2. Check advanced_evaluation_results.png for evaluation metrics")
            print(f"3. Run: streamlit run app.py")
            print(f"4. Upload retinal images to test the model")
        else:
            print("Training failed - please check error messages above")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()