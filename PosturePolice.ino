// Include the Servo library 
#include <Servo.h>
// Declare the Servo pin 
const int servoPin = 3;

// Create a servo object 
Servo ServoArm; 

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  ServoArm.attach(servoPin);
}

void loop() {
  // put your main code here, to run repeatedly:
    if (Serial.available())
    {
      if (Serial.readString().equals("BP"))  //if bad posture
      {
        // swing arm around 
        // Make servo go to 0 degrees 
        ServoArm.write(0); 
        delay(1000); 
        // Make servo go to 90 degrees 
        ServoArm.write(90); 
        delay(1000); 
        // Make servo go to 180 degrees 
        ServoArm.write(180);
      }
      delay(500);
    }    
    
}




