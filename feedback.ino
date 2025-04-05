int feedbackPin = 13; // LED connected to pin 13

void setup() {
  pinMode(feedbackPin, OUTPUT);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    char input = Serial.read();
    if(input == '1') {
      digitalWrite(feedbackPin, HIGH);
      delay(1000);
      digitalWrite(feedbackPin, LOW);
    }
  }
}
