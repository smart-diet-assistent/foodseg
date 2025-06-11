#!/usr/bin/env python3
"""
Convert ONNX model to TensorFlow Lite format for Arduino deployment
"""

import onnx
import tensorflow as tf
from onnx_tf.backend import prepare
import numpy as np
import os

def convert_onnx_to_tflite(onnx_path, tflite_path):
    """
    Convert ONNX model to TensorFlow Lite format
    
    Args:
        onnx_path (str): Path to input ONNX model
        tflite_path (str): Path to output TFLite model
    """
    print(f"Loading ONNX model from: {onnx_path}")
    
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX to TensorFlow
    print("Converting ONNX to TensorFlow...")
    tf_rep = prepare(onnx_model)
    
    # Export to saved model format
    saved_model_path = tflite_path.replace('.tflite', '_saved_model')
    tf_rep.export_graph(saved_model_path)
    
    # Convert to TensorFlow Lite
    print("Converting TensorFlow to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
    
    # Optimize for size (important for microcontrollers)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # For even smaller models, you can use:
    # converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TensorFlow Lite model saved to: {tflite_path}")
    
    # Print model info
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("\nModel Information:")
    print(f"Model size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Input type: {input_details[0]['dtype']}")
    print(f"Output shape: {output_details[0]['shape']}")
    print(f"Output type: {output_details[0]['dtype']}")
    
    return tflite_path

def generate_model_data_cc(tflite_path, output_path):
    """
    Generate C++ array file from TFLite model for Arduino
    """
    print(f"Generating C++ array from: {tflite_path}")
    
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Generate C++ header
    cpp_code = f'''// Auto-generated model data for Arduino
// Generated from: {os.path.basename(tflite_path)}

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

const unsigned char g_model[] = {{
'''
    
    # Add hex data
    hex_array = []
    for i, byte in enumerate(model_data):
        if i % 16 == 0:
            hex_array.append('\n  ')
        hex_array.append(f'0x{byte:02x}')
        if i < len(model_data) - 1:
            hex_array.append(', ')
    
    cpp_code += ''.join(hex_array)
    cpp_code += f'''
}};

const int g_model_len = {len(model_data)};

#endif  // MODEL_DATA_H
'''
    
    with open(output_path, 'w') as f:
        f.write(cpp_code)
    
    print(f"C++ model data saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    # Paths
    onnx_model_path = "models/food_segmentation_model.onnx"
    tflite_model_path = "models/food_segmentation_model.tflite"
    cpp_model_path = "esp32_deployment/model_data.cc"
    
    # Check if ONNX model exists
    if not os.path.exists(onnx_model_path):
        print(f"Error: ONNX model not found at {onnx_model_path}")
        exit(1)
    
    try:
        # Convert ONNX to TFLite
        convert_onnx_to_tflite(onnx_model_path, tflite_model_path)
        
        # Generate C++ array
        generate_model_data_cc(tflite_model_path, cpp_model_path)
        
        print("\nConversion completed successfully!")
        print(f"TensorFlow Lite model: {tflite_model_path}")
        print(f"C++ model data: {cpp_model_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        print("Make sure you have the required dependencies:")
        print("pip install onnx onnx-tf tensorflow")