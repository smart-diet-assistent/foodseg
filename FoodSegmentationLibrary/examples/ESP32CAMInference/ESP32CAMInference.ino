/*
  ESP32-CAM Food Segmentation Example
  
  This example demonstrates how to use the FoodSegmentationLibrary
  with an ESP32-CAM to capture and analyze food images in real-time.
  
  Features:
  - Real-time camera capture
  - Food segmentation inference
  - Web server for viewing results
  - JSON API for integration
  
  Hardware requirements:
  - ESP32-CAM board
  - 4MB PSRAM (recommended)
  - MicroSD card (optional, for saving images)
  
  Created by Food Segmentation Team
  
  This example code is in the public domain.
*/

#include "FoodSegmentation.h"
#include "esp_camera.h"
#include "WiFi.h"
#include "WebServer.h"
#include "ArduinoJson.h"
#include "FS.h"
#include "SD_MMC.h"

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Camera pins for ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Flash LED pin
#define FLASH_LED_PIN      4

// Create instances
FoodSegmentation foodSeg;
WebServer server(80);

// Buffers
float* input_buffer;
uint8_t* output_mask;
uint8_t* colored_mask;
uint8_t* resized_buffer;

// Status variables
bool camera_initialized = false;
bool model_initialized = false;
unsigned long last_inference = 0;
const unsigned long inference_interval = 2000; // Run inference every 2 seconds

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  
  Serial.println("ESP32-CAM Food Segmentation System");
  Serial.println("==================================");
  
  // Initialize flash LED
  pinMode(FLASH_LED_PIN, OUTPUT);
  digitalWrite(FLASH_LED_PIN, LOW);
  
  // Initialize camera
  if (initCamera()) {
    Serial.println("✅ Camera initialized");
    camera_initialized = true;
  } else {
    Serial.println("❌ Camera initialization failed");
  }
  
  // Allocate memory buffers
  if (allocateBuffers()) {
    Serial.println("✅ Buffers allocated");
  } else {
    Serial.println("❌ Buffer allocation failed");
    return;
  }
  
  // Initialize food segmentation model
  if (foodSeg.begin()) {
    Serial.println("✅ Food segmentation model initialized");
    model_initialized = true;
  } else {
    Serial.print("❌ Model initialization failed: ");
    Serial.println(foodSeg.getLastError());
  }
  
  // Connect to WiFi
  connectToWiFi();
  
  // Setup web server routes
  setupWebServer();
  
  // Initialize SD card (optional)
  initSDCard();
  
  Serial.println("🚀 System ready!");
  if (camera_initialized && model_initialized) {
    Serial.println("📸 Ready for food segmentation");
    printInstructions();
  }
}

void loop() {
  server.handleClient();
  
  // Run automatic inference if enabled
  if (camera_initialized && model_initialized) {
    unsigned long now = millis();
    if (now - last_inference > inference_interval) {
      runAutomaticInference();
      last_inference = now;
    }
  }
  
  delay(10);
}

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_RGB565;
  
  // Higher resolution for better quality
  config.frame_size = FRAMESIZE_QVGA; // 320x240
  config.jpeg_quality = 12;
  config.fb_count = 1;
  
  // Camera init
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return false;
  }
  
  return true;
}

bool allocateBuffers() {
  input_buffer = (float*)malloc(MODEL_INPUT_SIZE * sizeof(float));
  output_mask = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT);
  colored_mask = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3);
  resized_buffer = (uint8_t*)malloc(MODEL_INPUT_WIDTH * MODEL_INPUT_HEIGHT * 3);
  
  return (input_buffer && output_mask && colored_mask && resized_buffer);
}

void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println();
  Serial.print("✅ WiFi connected! IP address: ");
  Serial.println(WiFi.localIP());
}

void setupWebServer() {
  // Home page
  server.on("/", handleRoot);
  
  // Capture and analyze image
  server.on("/capture", handleCapture);
  
  // Get latest analysis results as JSON
  server.on("/results", handleResults);
  
  // Stream current camera image
  server.on("/stream", handleStream);
  
  // Control flash LED
  server.on("/flash", handleFlash);
  
  server.begin();
  Serial.println("🌐 Web server started");
}

void handleRoot() {
  String html = R"(
<!DOCTYPE html>
<html>
<head>
    <title>ESP32-CAM Food Segmentation</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        button { padding: 10px 20px; margin: 10px; font-size: 16px; }
        .results { margin-top: 20px; padding: 15px; border: 1px solid #ccc; }
        .image-container { margin: 10px 0; }
        img { max-width: 100%; height: auto; border: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🍽️ ESP32-CAM Food Segmentation</h1>
        
        <div>
            <button onclick="captureAndAnalyze()">📸 Capture & Analyze</button>
            <button onclick="toggleFlash()">💡 Toggle Flash</button>
            <button onclick="getResults()">📊 Get Results</button>
        </div>
        
        <div class="image-container">
            <h3>Camera Stream:</h3>
            <img id="stream" src="/stream" alt="Camera Stream">
        </div>
        
        <div id="results" class="results">
            <h3>Analysis Results:</h3>
            <p>Click "Capture & Analyze" to start</p>
        </div>
    </div>
    
    <script>
        function captureAndAnalyze() {
            fetch('/capture')
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('results').innerHTML = '<p>Error: ' + error + '</p>';
                });
        }
        
        function getResults() {
            fetch('/results')
                .then(response => response.json())
                .then(data => {
                    displayResults(data);
                })
                .catch(error => console.error('Error:', error));
        }
        
        function toggleFlash() {
            fetch('/flash');
        }
        
        function displayResults(data) {
            let html = '<h3>Analysis Results:</h3>';
            
            if (data.success) {
                html += '<p><strong>Inference Time:</strong> ' + data.inference_time + ' ms</p>';
                html += '<p><strong>Dominant Food:</strong> ' + data.dominant_food + ' (' + data.dominant_percentage + '%)</p>';
                
                html += '<h4>Detected Foods:</h4><ul>';
                data.detected_foods.forEach(food => {
                    html += '<li>' + food.name + ': ' + food.percentage + '%</li>';
                });
                html += '</ul>';
            } else {
                html += '<p><strong>Error:</strong> ' + data.error + '</p>';
            }
            
            document.getElementById('results').innerHTML = html;
        }
        
        // Auto-refresh stream
        setInterval(() => {
            document.getElementById('stream').src = '/stream?' + Date.now();
        }, 2000);
    </script>
</body>
</html>
  )";
  
  server.send(200, "text/html", html);
}

void handleCapture() {
  if (!camera_initialized || !model_initialized) {
    server.send(500, "application/json", "{\"success\":false,\"error\":\"System not initialized\"}");
    return;
  }
  
  // Capture image and run inference
  JsonObject result = runInference();
  
  String response;
  serializeJson(result, response);
  server.send(200, "application/json", response);
}

void handleResults() {
  // Return cached results or run new inference
  JsonObject result = runInference();
  
  String response;
  serializeJson(result, response);
  server.send(200, "application/json", response);
}

void handleStream() {
  if (!camera_initialized) {
    server.send(404, "text/plain", "Camera not available");
    return;
  }
  
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Camera capture failed");
    return;
  }
  
  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Length", String(fb->len));
  server.sendHeader("Cache-Control", "no-cache");
  server.send_P(200, "image/jpeg", (const char*)fb->buf, fb->len);
  
  esp_camera_fb_return(fb);
}

void handleFlash() {
  static bool flash_on = false;
  flash_on = !flash_on;
  digitalWrite(FLASH_LED_PIN, flash_on ? HIGH : LOW);
  server.send(200, "text/plain", flash_on ? "Flash ON" : "Flash OFF");
}

JsonObject runInference() {
  DynamicJsonDocument doc(2048);
  JsonObject result = doc.to<JsonObject>();
  
  unsigned long start_time = millis();
  
  // Capture image from camera
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    result["success"] = false;
    result["error"] = "Camera capture failed";
    return result;
  }
  
  // Convert and resize image to model input format
  if (!convertCameraImage(fb)) {
    esp_camera_fb_return(fb);
    result["success"] = false;
    result["error"] = "Image conversion failed";
    return result;
  }
  
  esp_camera_fb_return(fb);
  
  // Preprocess image
  if (!foodSeg.preprocessImage(resized_buffer, input_buffer)) {
    result["success"] = false;
    result["error"] = "Image preprocessing failed";
    return result;
  }
  
  // Run inference
  if (!foodSeg.predict(input_buffer, output_mask)) {
    result["success"] = false;
    result["error"] = foodSeg.getLastError();
    return result;
  }
  
  unsigned long inference_time = millis() - start_time;
  
  // Analyze results
  float class_areas[NUM_FOOD_CLASSES];
  foodSeg.calculateClassAreas(output_mask, class_areas);
  
  float max_percentage;
  int dominant_class = foodSeg.getClassWithMaxArea(output_mask, &max_percentage);
  
  // Build response
  result["success"] = true;
  result["inference_time"] = inference_time;
  result["dominant_food"] = foodSeg.getClassName(dominant_class);
  result["dominant_percentage"] = round(max_percentage * 10) / 10.0;
  
  JsonArray detected_foods = result.createNestedArray("detected_foods");
  for (int i = 1; i < NUM_FOOD_CLASSES; i++) {
    if (class_areas[i] > 1.0) {
      JsonObject food = detected_foods.createNestedObject();
      food["name"] = foodSeg.getClassName(i);
      food["percentage"] = round(class_areas[i] * 10) / 10.0;
    }
  }
  
  return result;
}

bool convertCameraImage(camera_fb_t* fb) {
  // Convert RGB565 to RGB888 and resize to model input size
  // This is a simplified conversion - in production you might want more sophisticated processing
  
  if (fb->format != PIXFORMAT_RGB565) {
    return false;
  }
  
  // Simple resize and conversion
  uint16_t* rgb565_data = (uint16_t*)fb->buf;
  int src_width = fb->width;
  int src_height = fb->height;
  
  float x_ratio = (float)src_width / MODEL_INPUT_WIDTH;
  float y_ratio = (float)src_height / MODEL_INPUT_HEIGHT;
  
  for (int y = 0; y < MODEL_INPUT_HEIGHT; y++) {
    for (int x = 0; x < MODEL_INPUT_WIDTH; x++) {
      int src_x = (int)(x * x_ratio);
      int src_y = (int)(y * y_ratio);
      
      src_x = min(src_x, src_width - 1);
      src_y = min(src_y, src_height - 1);
      
      uint16_t pixel = rgb565_data[src_y * src_width + src_x];
      
      // Convert RGB565 to RGB888
      uint8_t r = (pixel >> 11) << 3;
      uint8_t g = ((pixel >> 5) & 0x3F) << 2;
      uint8_t b = (pixel & 0x1F) << 3;
      
      int dst_idx = (y * MODEL_INPUT_WIDTH + x) * 3;
      resized_buffer[dst_idx] = r;
      resized_buffer[dst_idx + 1] = g;
      resized_buffer[dst_idx + 2] = b;
    }
  }
  
  return true;
}

void runAutomaticInference() {
  // Run inference silently for continuous monitoring
  JsonObject result = runInference();
  
  if (result["success"]) {
    Serial.print("🍽️ Detected: ");
    Serial.print(result["dominant_food"].as<String>());
    Serial.print(" (");
    Serial.print(result["dominant_percentage"].as<float>());
    Serial.println("%)");
  }
}

void initSDCard() {
  if (!SD_MMC.begin()) {
    Serial.println("⚠️ SD Card initialization failed");
    return;
  }
  
  Serial.println("✅ SD Card initialized");
}

void printInstructions() {
  Serial.println("\n📋 Instructions:");
  Serial.println("================");
  Serial.println("1. Open web browser and go to: http://" + WiFi.localIP().toString());
  Serial.println("2. Click 'Capture & Analyze' to take a photo and analyze food");
  Serial.println("3. Use 'Toggle Flash' to control LED illumination");
  Serial.println("4. The system also runs automatic inference every 2 seconds");
  Serial.println("\n📡 API Endpoints:");
  Serial.println("- GET / : Web interface");
  Serial.println("- GET /capture : Capture image and analyze (JSON response)");
  Serial.println("- GET /results : Get latest analysis results (JSON response)");
  Serial.println("- GET /stream : Camera stream image");
  Serial.println("- GET /flash : Toggle flash LED");
}
