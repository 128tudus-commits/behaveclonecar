#include <Arduino.h>
#include <Servo.h>
#include <stdlib.h>

const uint8_t STEERING_SERVO_PIN = 9;
const uint8_t THROTTLE_PWM_PIN = 5;

const uint16_t SERVO_MIN_US = 1000;
const uint16_t SERVO_MAX_US = 2000;
const int16_t SERVO_TRIM_US = 0;

const uint8_t THROTTLE_RESOLUTION = 255;

const unsigned long FAILSAFE_TIMEOUT_MS = 500;
const bool FAILSAFE_ENABLED = true;

Servo steeringServo;
char rxBuffer[64];

void applySteering(float steering)
{
    steering = constrain(steering, -1.0f, 1.0f);
    float fraction = (steering + 1.0f) / 2.0f;
    uint16_t pulseUs = SERVO_MIN_US
                     + (uint16_t)(fraction * (SERVO_MAX_US - SERVO_MIN_US))
                     + SERVO_TRIM_US;
    
    steeringServo.writeMicroseconds(pulseUs);
}

void applyThrottle(float throttle)
{
    throttle = constrain(throttle, 0.0f, 1.0f);
    analogWrite(THROTTLE_PWM_PIN, (uint8_t)(throttle * THROTTLE_RESOLUTION + 0.5f));
}

bool parseControls(const char* line, float& outSteering, float& outThrottle)
{
    
    const char* ptr = strchr(line, '*');
    if (!ptr) return false;
    ptr++; 
void applyNeutral()
{
    applySteering(0.0f);
    applyThrottle(0.0f);
}

    
    char* nextPtr;
    float parsedSteering = strtod(ptr, &nextPtr);
    if (ptr == nextPtr) return false; // Nie udało się odczytać liczby
    
    
    ptr = strchr(nextPtr, ',');
    if (!ptr) return false;
    
  
    ptr = strchr(ptr, '*');
    if (!ptr) return false;
    ptr++;
    
    float parsedThrottle = strtod(ptr, &nextPtr);
    if (ptr == nextPtr) return false; // Nie udało się odczytać liczby
    
    outSteering = parsedSteering;
    outThrottle = parsedThrottle;
    return true;
}

void setup()
{
    Serial.begin(115200);

    steeringServo.attach(STEERING_SERVO_PIN, SERVO_MIN_US, SERVO_MAX_US);

    pinMode(THROTTLE_PWM_PIN, OUTPUT);
    applyNeutral();
}

void loop()
{
    static uint8_t rxIndex = 0;
    static unsigned long lastCommandMs = 0;
    static bool failsafeActive = true;

    while (Serial.available() > 0) {
        char c = (char)Serial.read();

        if (c == '\n') {
            rxBuffer[rxIndex] = '\0';
            
            float steering = 0.0f;
            float throttle = 0.0f;

            if (parseControls(rxBuffer, steering, throttle)) {
                applySteering(steering);
                applyThrottle(throttle);
                lastCommandMs = millis();
                failsafeActive = false;
            }

            rxIndex = 0;
        } else if (c != '\r') {
            if (rxIndex < 63) {
                rxBuffer[rxIndex++] = c;
            } else {
                rxIndex = 0;
            }
        }
    }

    if (FAILSAFE_ENABLED && !failsafeActive && (millis() - lastCommandMs > FAILSAFE_TIMEOUT_MS)) {
        applyNeutral();
        failsafeActive = true;
    }
}
