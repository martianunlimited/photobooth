const int footSwitchPin = 2;  // Pin where the foot switch is connected
int footSwitchState = 0;      // Variable for reading the foot switch status

void setup() {
  pinMode(footSwitchPin, INPUT_PULLUP);  // Set the pin to input with internal pull-up resistor
  Serial.begin(9600);  // Start the serial communication
}

void loop() {
  footSwitchState = digitalRead(footSwitchPin);  // Read the foot switch state

  // When the foot switch is pressed (LOW)
  if (footSwitchState == LOW) {
    delay(50);  // Debounce the switch
    if (digitalRead(footSwitchPin) == LOW) {
      Serial.println("footSwitchPressed");  // Send a message via serial
      delay(500);  // Prevent multiple sends if the button is held down
    }
  }
}
