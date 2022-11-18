#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <LiquidCrystal.h>
#include<ADS1115_WE.h> 
#include<Wire.h>
#define I2C_ADDRESS 0x48

// LCD
const int rs = D7, en = D6, d4 = D5, d5 = D4, d6 = D3, d7 = D0;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

// ADC
ADS1115_WE adc = ADS1115_WE(I2C_ADDRESS);

// Internet Connection
WiFiClient espClient;
// Name of your Wifi
const char* ssid = "";
// Password for your Wifi 
const char* password = "";

// MQTT
PubSubClient client(espClient);
// Host Laptop IP Address - Using IP Config
const char* mqtt_server = "192.168.43.172";

#define MSG_BUFFER_SIZE  (50)
char msg[MSG_BUFFER_SIZE];

#define LEFT 1
#define RIGHT 2
int dir = -1; //-1 for neutral, 1 up, 2 down, 3 left, 4 right
bool pending_result = false;
bool received_char = false;

void setup_wifi() {

  delay(10);
  // We start by connecting to a WiFi network
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  randomSeed(micros());

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

// MQTT Callback - Called when model classify an alphabet successfully
void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");
  lcd.setCursor(0, 1);
  lcd.print("                 ");
  lcd.setCursor(0, 1);
  pending_result = false; //once received result, no longer waiting for result
  
  for (int i = 0; i < length; i++) {
    Serial.print((char)payload[i]);
    lcd.print((char) payload[i]);
  }
  Serial.println();
  received_char = true;
}

void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(), "weiming", "weiming")) {
      Serial.println("connected");
      // Once connected, publish an announcement...
      // ... and resubscribe
      client.subscribe("Group_25/NLP/Output"); //listen to output from NLP Client
    } else {
      Serial.print(client.connect(clientId.c_str()));
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      delay(5000);
    }
  }
}

void setup() {
  lcd.begin(16, 2);
  lcd.setCursor(0, 0);
  lcd.print("Connecting Wifi");
  //Wifi Setup
  setup_wifi();
  
  Wire.begin();
  Serial.begin(115200);
  
  lcd.setCursor(0, 0);
  lcd.print("               ");
  lcd.setCursor(0, 0);
  // MQTT Setup
  lcd.print("Connecting MQTT");
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  lcd.setCursor(0, 0);
  lcd.print("               ");
  lcd.setCursor(0, 0);
  lcd.print("Toggle Right");

  // ADC Setup
  if(!adc.init())Serial.println("ADS1115 not connected!");
  adc.setVoltageRange_mV(ADS1115_RANGE_6144);
  adc.setCompareChannels(ADS1115_COMP_0_GND);
  adc.setMeasureMode(ADS1115_CONTINUOUS); 
}
void loop() {

  if (!client.connected()) {
    reconnect();
  }
  
  client.loop();

  dir = getDirection(); //checks joystick input
  
  //If a right is detected while classification is underway, sends a space to NLP client
  if (pending_result && dir == RIGHT){
    client.publish("Group_25/NLP/Input", "space"); // sending "space" to denote space
    delay(250); 
  }
  //Else if classification is not started, a right means to start classification
  else if (pending_result == false && dir == RIGHT){
    pending_result = true;
  }
  //If a left is detected, it means the NLP Client should output the sentence back to Arduino and the Text to Speech App
  if (dir == LEFT){
    client.publish("Group_25/NLP/Input", "output");
    delay(250);
  }
  
  //If classification is underway, send a "1" to the Image Receiver to continue sending image to the CNN Client
  if (pending_result){
    lcd.setCursor(0, 0);
    lcd.print("                 ");
    lcd.setCursor(0, 0);
    lcd.print("Classifying");
    client.publish("Group_25/ImageRec/Notify", "1"); // sending a "1" to the ip camera
    delay(250); //to prevent spamming too much image
  }

  // Once a character is received, stop sending "1" to Image Receiver until the next classification command is given.
  if (received_char){
    // ask user to press button again to read another character
    lcd.setCursor(0, 0);
    lcd.print("               ");
    lcd.setCursor(0, 0);
    lcd.print("Toggle Right");
    pending_result = false;
    received_char = false;
  }
}

// Gets the direction from the Joystick
int getDirection(){
  float Va0;
  Va0 = readChannel(ADS1115_COMP_0_GND);

  if (Va0 > 4){
    dir = RIGHT;
    Serial.println("Right");
  }
  else if (Va0 < 1) {
    dir = LEFT;
    Serial.println("Left");
   
  }
  else{
    dir = -1;
    Serial.println("Neutral");
  }
  return dir;
  
}

// Converting analog reading to voltage values
float readChannel(ADS1115_MUX channel) {
  float voltage = 0.0;
  adc.setCompareChannels(channel);
  voltage = adc.getResult_V(); // alternative: getResult_mV for Millivolt
  return voltage;
}
