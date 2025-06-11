# Food Segmentation Library for Arduino

A powerful Arduino library for real-time food segmentation using TensorFlow Lite. This library enables ESP32 and other Arduino-compatible boards to identify and segment 33 different food categories in real-time.

## Features

- 🍽️ **33 Food Categories**: Recognizes a wide variety of foods including fruits, vegetables, meats, grains, and more
- 🚀 **Real-time Inference**: Optimized for microcontrollers with fast inference times
- 📱 **ESP32-CAM Support**: Built-in support for ESP32-CAM modules with web interface
- 🎨 **Visualization**: Built-in color mapping for segmentation visualization
- 📊 **Analysis Tools**: Calculate food areas, percentages, and dominant food types
- 💾 **Memory Efficient**: Optimized for devices with limited memory
- 🌐 **IoT Ready**: Web server and JSON API support for integration

## Supported Food Categories

1. **Fruits**: Apple, Banana, Strawberry, Mango, Orange
2. **Vegetables**: Lettuce, Cucumber, Carrot, Cabbage, Green Bean, Eggplant, Potato, Garlic, Tomato, Scallion, Ginger
3. **Proteins**: Beef Steak, Pork, Chicken/Duck, Egg, Tofu
4. **Grains & Starches**: Bread, Corn, Pasta, Noodles, Rice
5. **Dairy**: Milk, Cheese/Butter
6. **Others**: Chocolate, Sauce, King Oyster Mushroom, Shiitake Mushroom

## Hardware Requirements

### Minimum Requirements
- ESP32 with 4MB Flash
- 320KB RAM (recommended: 4MB PSRAM)
- Camera module (for image capture)

### Recommended Hardware
- ESP32-CAM module
- 4MB PSRAM
- MicroSD card (optional, for saving images)
- External LED for flash (optional)

## Installation

### Arduino IDE
1. Download the library as a ZIP file
2. Open Arduino IDE
3. Go to **Sketch** → **Include Library** → **Add .ZIP Library**
4. Select the downloaded ZIP file
5. The library will be installed automatically

### Dependencies
Make sure to install these required libraries:
- **Arduino_TensorFlowLite** (ESP32 TensorFlow Lite library)
- **ArduinoJson** (for ESP32-CAM web interface)
- **WiFi** (built-in ESP32 library)

## Quick Start

### Basic Usage

```cpp
#include "FoodSegmentation.h"

FoodSegmentation foodSeg;
float* input_buffer;
uint8_t* output_mask;

void setup() {
  Serial.begin(115200);
  
  // Allocate memory
  input_buffer = (float*)malloc(MODEL_INPUT_SIZE * sizeof(float));
  output_mask = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT);
  
  // Initialize model
  if (!foodSeg.begin()) {
    Serial.println("Model initialization failed!");
    return;
  }
  
  Serial.println("Food segmentation ready!");
}

void loop() {
  // Your RGB image data (320x240x3)
  uint8_t rgb_image[MODEL_INPUT_SIZE];
  
  // Preprocess image
  foodSeg.preprocessImage(rgb_image, input_buffer);
  
  // Run inference
  if (foodSeg.predict(input_buffer, output_mask)) {
    // Get dominant food
    float percentage;
    int dominant_class = foodSeg.getClassWithMaxArea(output_mask, &percentage);
    
    Serial.print("Detected: ");
    Serial.print(foodSeg.getClassName(dominant_class));
    Serial.print(" (");
    Serial.print(percentage);
    Serial.println("%)");
  }
  
  delay(1000);
}
```

### ESP32-CAM Web Interface

The library includes a complete ESP32-CAM example with web interface:

1. Flash the `ESP32CAMInference` example to your ESP32-CAM
2. Configure WiFi credentials in the code
3. Open the web interface at the ESP32's IP address
4. Capture and analyze food images through the web browser

## API Reference

### FoodSegmentation Class

#### Constructor
```cpp
FoodSegmentation();
```

#### Methods

##### `bool begin()`
Initialize the TensorFlow Lite model and allocate resources.
- **Returns**: `true` if successful, `false` otherwise

##### `bool predict(float* input_image, uint8_t* output_mask)`
Run food segmentation inference on preprocessed image.
- **Parameters**:
  - `input_image`: Normalized float array (320×240×3)
  - `output_mask`: Output segmentation mask (320×240)
- **Returns**: `true` if successful, `false` otherwise

##### `bool preprocessImage(uint8_t* rgb_image, float* normalized_output)`
Convert RGB image to normalized input format.
- **Parameters**:
  - `rgb_image`: Input RGB image (320×240×3)
  - `normalized_output`: Output normalized array
- **Returns**: `true` if successful, `false` otherwise

##### `int getClassWithMaxArea(uint8_t* mask, float* percentage = nullptr)`
Find the food class with the largest area in the segmentation mask.
- **Parameters**:
  - `mask`: Segmentation mask
  - `percentage`: Optional output for percentage coverage
- **Returns**: Class ID of dominant food

##### `void calculateClassAreas(uint8_t* mask, float* areas)`
Calculate area percentages for all food classes.
- **Parameters**:
  - `mask`: Segmentation mask
  - `areas`: Output array for class percentages

##### `const char* getClassName(int class_id)`
Get human-readable name for a class ID.
- **Parameters**:
  - `class_id`: Food class ID (0-32)
- **Returns**: Class name string

##### `bool isInitialized()`
Check if the model is properly initialized.
- **Returns**: `true` if initialized, `false` otherwise

##### `const char* getLastError()`
Get the last error message.
- **Returns**: Error message string

### Utility Functions

#### `FoodSegmentationUtils::rgbToFloat()`
Convert RGB image to normalized float format.

#### `FoodSegmentationUtils::resizeImage()`
Resize image using nearest neighbor interpolation.

#### `FoodSegmentationUtils::applyColorMap()`
Apply color mapping to segmentation mask for visualization.

#### `FoodSegmentationUtils::calculateIoU()`
Calculate Intersection over Union between two masks.

## Examples

### 1. Basic Inference
Simple example showing how to use the library for food segmentation.

### 2. ESP32-CAM Inference
Complete ESP32-CAM implementation with:
- Real-time camera capture
- Web interface
- JSON API
- Flash LED control
- Automatic inference

## Memory Usage

- **Model Size**: ~3.7MB (stored in flash)
- **Runtime Memory**: ~600KB (for inference)
- **Input Buffer**: ~921KB (320×240×3×4 bytes)
- **Output Buffer**: ~77KB (320×240 bytes)

**Total RAM Usage**: ~1.6MB (requires ESP32 with PSRAM)

## Performance

- **Inference Time**: 200-500ms (depending on ESP32 variant)
- **Input Resolution**: 320×240 pixels
- **Output Resolution**: 320×240 pixels
- **Accuracy**: >85% mIoU on test dataset

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Ensure your ESP32 has PSRAM enabled
   - Check that PSRAM is properly configured in Arduino IDE

2. **Model Initialization Failed**
   - Verify that the model data is properly included
   - Check available flash memory

3. **Camera Initialization Failed**
   - Verify camera connections
   - Check camera pin configuration

4. **Poor Recognition Accuracy**
   - Ensure good lighting conditions
   - Check that food items are clearly visible
   - Verify camera focus

### Debug Tips

Enable debug output:
```cpp
Serial.println(foodSeg.getLastError());
```

Check memory usage:
```cpp
Serial.print("Free heap: ");
Serial.println(ESP.getFreeHeap());
```

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This library is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- TensorFlow Lite team for the micro framework
- ESP32 community for hardware support
- Food dataset contributors for training data

## Version History

### v1.0.0
- Initial release
- 33 food category support
- ESP32-CAM integration
- Web interface
- Basic inference examples
