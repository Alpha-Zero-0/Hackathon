// Include the Servo library 
 #include <Servo.h>
 // Declare the Servo pin 
 const int servoPin = 3;
 
 // Buzzer pin
 const int buzzerPin = 8; 
 
 // Create a servo object 
 Servo ServoArm; 
 
 void setup() {
   // put your setup code here, to run once:
   Serial.begin(9600);
   pinMode(buzzerPin,OUTPUT);
   ServoArm.detach();
   ServoArm.attach(servoPin);
 }
 
 void loop() {
   // put your main code here, to run repeatedly:
     if (Serial.available() > 0)
     {
       String msg = Serial.readStringUntil('\n');
       msg.trim();
       if (msg=="BP")  //if bad posture
       {
         // swing arm around 
         // Make servo go to 40 degrees 
         ServoArm.write(40); 
         delay(1000); 
         // Make servo go to 90 degrees 
         ServoArm.write(90); 
         delay(1000); 
       }
       else if (msg=="GP")
       {
        ServoArm.write(90);
        delay(1000);
       }
       else
       {
        ServoArm.write(90);
        delay(1000);
       }
     }
 
 }

