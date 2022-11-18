//KY018 Photo resistor module
 
int sensorPin = A0; // select the input pin for the potentiometer
int ledPin = D3; // select the pin for the LED
int sensorValue = 0; // variable to store the value coming from the sensor
void setup() {
  pinMode(ledPin, OUTPUT);
  pinMode(sensorPin, INPUT);
  Serial.begin(9600);
}
void loop() {
  sensorValue = analogRead(sensorPin);
  Serial.println(sensorValue);
  if (sensorValue<1000){
    digitalWrite(ledPin,LOW);
  } else{
    digitalWrite(ledPin,HIGH);
  }
}
