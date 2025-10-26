"""
GPU Configuration for TensorFlow with 2GB GPU
This script configures TensorFlow to work efficiently with limited GPU memory
"""
import tensorflow as tf
import os

def configure_gpu():
    """
    Configure GPU settings for TensorFlow to work with 2GB GPU
    """
    # Check if GPU is available
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth - this is CRITICAL for 2GB GPU
            # Instead of allocating all GPU memory at once, TensorFlow will allocate as needed
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Optional: Set a memory limit (e.g., 1.5GB out of 2GB to leave room for system)
            # Uncomment if you want to set a hard limit
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=1536)])  # 1.5GB
            
            print(f"✓ GPU configured successfully: {len(gpus)} GPU(s) available")
            print(f"  GPU Name: {gpus[0].name}")
            
            # Enable mixed precision for better performance and memory efficiency
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("✓ Mixed precision enabled (float16)")
            
        except RuntimeError as e:
            print(f"✗ GPU configuration error: {e}")
    else:
        print("✗ No GPU found. Running on CPU.")
        print("  This will be slower but will still work.")
    
    return gpus

def get_gpu_info():
    """
    Display GPU information
    """
    print("\n=== GPU Information ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
    else:
        print("No GPUs available - using CPU")
    
    print("=" * 30)

if __name__ == "__main__":
    configure_gpu()
    get_gpu_info()
    
    # Test GPU
    print("\n=== Testing GPU ===")
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"Test computation result:\n{c.numpy()}")
    print("✓ GPU test completed successfully!")
