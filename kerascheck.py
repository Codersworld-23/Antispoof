import importlib

def is_keras_installed():
    """Checks if Keras is installed."""
    try:
        importlib.import_module('keras')  # Check for standalone Keras
        print("Standalone Keras is installed.")
        return True
    except ImportError:
        try:
            importlib.import_module('tensorflow.keras')  # Check for TensorFlow Keras
            print("TensorFlow Keras is installed.")
            return True
        except ImportError:
            print("Keras is not installed.")
            return False

is_keras_installed()