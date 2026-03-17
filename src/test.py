import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")

try:
    from transformers import TFBertForSequenceClassification
    print("SUCCESS: Hugging Face found TensorFlow!")
except ImportError as e:
    print(f"FAILED: {e}")