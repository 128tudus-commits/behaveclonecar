import argparse
import sys
import time

try:
    import serial
except ImportError:
    serial = None

from record import Gamepad, detect_arduino_port


def parse_args():
    parser = argparse.ArgumentParser(description="Manual driving: gamepad -> Arduino (format: *steering*, *throttle*), no dataset recording")
    parser.add_argument("--pad", type=int, default=0, help="Gamepad index")
    parser.add_argument("--steer-axis", type=int, default=3, help="Steering axis")
    parser.add_argument("--gas-axis", type=int, default=1, help="Gas axis (left joystick Y)")
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("-p", "--port", default="", help="Arduino port (auto-detect when empty, 'none' disables sending)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--send-fps", type=float, default=20.0, help="Command send rate")
    return parser.parse_args()


def read_key():
    try:
        import msvcrt
        if msvcrt.kbhit():
            return msvcrt.getwch()
    except ImportError:
        pass
    return None


def main():
    args = parse_args()

    try:
        pad = Gamepad(args.pad, args.steer_axis, args.gas_axis, args.deadzone)
    except RuntimeError as e:
        print(e)
        sys.exit(1)

    port = args.port.strip()
    ser = None
    if port.lower() == "none":
        print("Sending to Arduino disabled (--port none) - preview mode.")
    else:
        if serial is None:
            print("Missing pyserial library. Install: pip install pyserial")
            pad.quit()
            sys.exit(1)
        if not port:
            print("Detecting Arduino port...")
            port = detect_arduino_port(args.baud) or ""
            print(f"Found Arduino: {port}" if port else "Arduino not found - preview mode.")
        else:
            print(f"Arduino: {port}")
        if port:
            try:
                ser = serial.Serial(port, args.baud, timeout=1)
            except Exception as e:
                print(f"Cannot open {port}: {e}")
                pad.quit()
                sys.exit(1)

    send_interval = 1.0 / max(args.send_fps, 0.1)
    last_send = 0.0

    print("Q or CTRL+C - quit")

    try:
        while True:
            steer, gas, _axes = pad.poll()

            now = time.time()
            if ser is not None and (now - last_send) >= send_interval:
                last_send = now
                cmd = f"*{steer:.3f}*, *{gas:.3f}*\n"
                try:
                    ser.write(cmd.encode("ascii"))
                except Exception as e:
                    print(f"\nSend error: {e}")
                    try:
                        ser.write(b"*0.000*, *0.000*\n")
                    except Exception:
                        pass
                    ser.close()
                    ser = None

            status = f"Steer: {steer:+.2f}  Gas: {gas:.2f}  {'[SENDING]' if ser is not None else '[PREVIEW]'}"
            print("\r" + status + "   ", end="", flush=True)

            key = read_key()
            if key and key.lower() == "q":
                break

            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        if ser is not None:
            try:
                ser.write(b"*0.000*, *0.000*\n")
            except Exception:
                pass
            ser.close()
        pad.quit()
        print("\nFinished.")


if __name__ == "__main__":
    main()

