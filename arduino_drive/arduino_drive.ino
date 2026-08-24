#include <Arduino.h>

const uint8_t STEERING_SERVO_PIN = 3;
const uint8_t THROTTLE_PWM_PIN = 9;

const uint16_t SERVO_MIN_US = 1000;
const uint16_t SERVO_CENTER_US = 1500;
const uint16_t SERVO_MAX_US = 2000;
const int16_t SERVO_TRIM_US = 0;

const uint8_t THROTTLE_RESOLUTION = 255;

const unsigned long FAILSAFE_TIMEOUT_MS = 500;
const bool FAILSAFE_ENABLED = true;

const uint16_t TICKS_PER_US = 2;
const uint16_t PULSE_LOW_TICKS = 38000;

volatile uint16_t g_pulseTicks = SERVO_CENTER_US * TICKS_PER_US;
volatile bool g_pulseHigh = false;

void setSteerPinHigh()
{
    PORTD |= _BV(PORTD0);
}

void setSteerPinLow()
{
    PORTD &= ~_BV(PORTD0);
}

ISR(TIMER3_COMPA_vect)
{
    if (g_pulseHigh) {
        setSteerPinLow();
        g_pulseHigh = false;
        OCR3A = PULSE_LOW_TICKS;
    } else {
        setSteerPinHigh();
        g_pulseHigh = true;
        OCR3A = g_pulseTicks;
    }
}

void setupSteeringTimer()
{
    pinMode(STEERING_SERVO_PIN, OUTPUT);
    digitalWrite(STEERING_SERVO_PIN, LOW);

    TCCR3A = 0;
    TCCR3B = _BV(WGM32) | _BV(CS31);
    TCNT3 = 0;
    OCR3A = PULSE_LOW_TICKS;
    TIMSK3 = _BV(OCIE3A);
}

void applySteering(float steering)
{
    steering = constrain(steering, -1.0f, 1.0f);
    float fraction = (steering + 1.0f) / 2.0f;
    uint16_t pulseUs = SERVO_MIN_US
                     + (uint16_t)(fraction * (SERVO_MAX_US - SERVO_MIN_US))
                     + SERVO_TRIM_US;
    noInterrupts();
    g_pulseTicks = pulseUs * TICKS_PER_US;
    interrupts();
}

void applyThrottle(float throttle)
{
    throttle = constrain(throttle, 0.0f, 1.0f);
    analogWrite(THROTTLE_PWM_PIN, (uint8_t)(throttle * THROTTLE_RESOLUTION + 0.5f));
}

void applyNeutral()
{
    applySteering(0.0f);
    applyThrottle(0.0f);
}

bool parseControls(const String& line, float& outSteering, float& outThrottle)
{
    int open1 = line.indexOf('*');
    if (open1 < 0) return false;

    int close1 = line.indexOf('*', open1 + 1);
    if (close1 < 0) return false;

    int separator = line.indexOf(',', close1 + 1);
    if (separator < 0) return false;

    int open2 = line.indexOf('*', separator + 1);
    if (open2 < 0) return false;

    int close2 = line.indexOf('*', open2 + 1);
    if (close2 < 0) return false;

    String steeringToken = line.substring(open1 + 1, close1);
    String throttleToken = line.substring(separator + 1, open2);

    steeringToken.trim();
    throttleToken.trim();

    if (steeringToken.length() == 0 || throttleToken.length() == 0) return false;

    outSteering = steeringToken.toFloat();
    outThrottle = throttleToken.toFloat();
    return true;
}

void setup()
{
    Serial.begin(115200);

    setupSteeringTimer();

    pinMode(THROTTLE_PWM_PIN, OUTPUT);
    applyNeutral();
}

void loop()
{
    static String rxBuffer = "";
    static unsigned long lastCommandMs = 0;
    static bool failsafeActive = true;

    while (Serial.available() > 0) {
        char c = (char)Serial.read();

        if (c == '\n') {
            float steering = 0.0f;
            float throttle = 0.0f;

            if (parseControls(rxBuffer, steering, throttle)) {
                applySteering(steering);
                applyThrottle(throttle);
                lastCommandMs = millis();
                failsafeActive = false;
            }

            rxBuffer = "";
        } else if (c != '\r') {
            if (rxBuffer.length() < 63) {
                rxBuffer += c;
            } else {
                rxBuffer = "";
            }
        }
    }

    if (FAILSAFE_ENABLED && !failsafeActive && (millis() - lastCommandMs > FAILSAFE_TIMEOUT_MS)) {
        applyNeutral();
        failsafeActive = true;
    }
}
