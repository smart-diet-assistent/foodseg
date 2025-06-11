/*
  Food Segmentation Basic Inference Example
  
  This example demonstrates how to use the FoodSegmentationLibrary
  to perform food segmentation on an ESP32-CAM or similar device.
  
  Hardware requirements:
  - ESP32-CAM or ESP32 with camera module
  - At least 4MB PSRAM recommended
  
  Created by Food Segmentation Team
  
  This example code is in the public domain.
*/

#include "FoodSegmentation.h"

// Create food segmentation instance
FoodSegmentation foodSeg;

// Buffer for model input (normalized float image)
float* input_buffer;

// Buffer for model output (segmentation mask)
uint8_t* output_mask;

// Buffer for colored visualization
uint8_t* colored_mask;

// Example RGB image data (320x240x3)
// In a real application, this would come from your camera
uint8_t example_image[MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3];

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  
  Serial.println("Food Segmentation Library - Basic Inference Example");
  Serial.println("===================================================");
  
  // Allocate memory for buffers
  input_buffer = (float*)malloc(MODEL_INPUT_SIZE * sizeof(float));
  output_mask = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT);
  colored_mask = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3);
  
  if (!input_buffer || !output_mask || !colored_mask) {
    Serial.println("❌ Failed to allocate memory for buffers");
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println("✅ Memory allocated successfully");
  
  // Initialize the food segmentation model
  Serial.println("🚀 Initializing food segmentation model...");
  
  if (!foodSeg.begin()) {
    Serial.print("❌ Failed to initialize model: ");
    Serial.println(foodSeg.getLastError());
    while (1) {
      delay(1000);
    }
  }
  
  Serial.println("✅ Model initialized successfully!");
  
  // Generate a simple test pattern for demonstration
  generateTestImage();
  
  Serial.println("📸 Test image generated");
  Serial.println("Ready for inference!");
}

void loop() {
  Serial.println("\n🔄 Running food segmentation inference...");
  
  unsigned long start_time = millis();
  
  // Step 1: Preprocess the image
  if (!foodSeg.preprocessImage(example_image, input_buffer)) {
    Serial.println("❌ Failed to preprocess image");
    delay(5000);
    return;
  }
  
  // Step 2: Run inference
  if (!foodSeg.predict(input_buffer, output_mask)) {
    Serial.print("❌ Inference failed: ");
    Serial.println(foodSeg.getLastError());
    delay(5000);
    return;
  }
  
  unsigned long inference_time = millis() - start_time;
  
  // Step 3: Analyze results
  analyzeResults();
  
  Serial.print("⏱️  Inference completed in ");
  Serial.print(inference_time);
  Serial.println(" ms");
  
  // Wait before next inference
  delay(10000);  // Run inference every 10 seconds
}

void generateTestImage() {
  // Generate a simple test pattern
  // In a real application, this would be replaced with camera capture
  
  Serial.println("🎨 Generating test pattern...");
  
  for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
    for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
      int idx = (y * MODEL_INPUT_WIDTH + x) * 3;
      
      // Create colored regions to simulate different foods
      if (x < MODEL_INPUT_WIDTH / 3) {
        // Left region - simulate apple (red)
        example_image[idx] = 200;      // R
        example_image[idx + 1] = 50;   // G
        example_image[idx + 2] = 50;   // B
      } else if (x < 2 * MODEL_INPUT_WIDTH / 3) {
        // Middle region - simulate banana (yellow)
        example_image[idx] = 255;      // R
        example_image[idx + 1] = 255;  // G
        example_image[idx + 2] = 0;    // B
      } else {
        // Right region - simulate lettuce (green)
        example_image[idx] = 50;       // R
        example_image[idx + 1] = 200;  // G
        example_image[idx + 2] = 50;   // B
      }
    }
  }
}

void analyzeResults() {
  Serial.println("\n📊 Analysis Results:");
  Serial.println("==================");
  
  // Calculate area percentages for each food class
  float class_areas[NUM_FOOD_CLASSES];
  foodSeg.calculateClassAreas(output_mask, class_areas);
  
  // Find the dominant food class
  float max_percentage;
  int dominant_class = foodSeg.getClassWithMaxArea(output_mask, &max_percentage);
  
  Serial.print("🥇 Dominant food: ");
  Serial.print(foodSeg.getClassName(dominant_class));
  Serial.print(" (");
  Serial.print(max_percentage, 1);
  Serial.println("%)");
  
  // Show all detected food categories with significant area
  Serial.println("\n🍽️  Detected Foods:");
  Serial.println("-------------------");
  
  for (int i = 1; i < NUM_FOOD_CLASSES; i++) {  // Skip background
    if (class_areas[i] > 1.0) {  // Only show foods with >1% area
      Serial.print("• ");
      Serial.print(foodSeg.getClassName(i));
      Serial.print(": ");
      Serial.print(class_areas[i], 1);
      Serial.println("%");
    }
  }
  
  // Generate colored visualization
  FoodSegmentationUtils::applyColorMap(output_mask, MODEL_INPUT_WIDTH, MODEL_INPUT_HEIGHT, colored_mask);
  
  Serial.println("🎨 Colored segmentation mask generated");
  
  // Print some pixel samples for debugging
  Serial.println("\n🔍 Sample pixels (first 10):");
  for (int i = 0; i < 10; i++) {
    Serial.print("Pixel ");
    Serial.print(i);
    Serial.print(": Class ");
    Serial.print(output_mask[i]);
    Serial.print(" (");
    Serial.print(foodSeg.getClassName(output_mask[i]));
    Serial.println(")");
  }
}

// Optional: Function to capture image from camera
// This would be implemented based on your specific camera module
/*
bool captureImageFromCamera(uint8_t* buffer) {
  // Implementation depends on your camera module
  // For ESP32-CAM, you would use the camera library
  // Return true if capture successful, false otherwise
  return false;
}
*/

// Optional: Function to save results to SD card
/*
void saveResultsToSD() {
  // Save segmentation mask and analysis results to SD card
  // Implementation depends on your SD card library
}
*/

// Optional: Function to send results via WiFi/Bluetooth
/*
void sendResultsWirelessly() {
  // Send analysis results via WiFi or Bluetooth
  // Could be used for IoT applications
}
*/
