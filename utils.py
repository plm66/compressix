import subprocess
import platform

def play_alert_sound():
    """Play a sound alert based on the operating system"""
    try:
        os_name = platform.system()
        
        if os_name == "Windows":
            print("\a")  # ASCII bell character
        elif os_name == "Darwin":  # macOS
            subprocess.call(["afplay", "/System/Library/Sounds/Ping.aiff"])
        else:  # Linux or other Unix
            print("\a")
            
        print("*** ALERT: Price breakout detected! ***")
    except Exception as e:
        print(f"*** ALERT: Price breakout detected! (Sound error: {e}) ***")