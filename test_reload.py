import os
import sys
import time

def main():
    print("Running main application logic...")
    # Place your main application logic here
    time.sleep(1)  # Simulate work by sleeping for 10 seconds

if __name__ == "__main__":
    while True:
        main()  # Run the main application logic
        print("Restarting application in 1 minute...")
        time.sleep(5)  # Wait for 1 minute
        print("Restarting now...")
        os.execv(sys.executable, ['python'] + sys.argv)  # Restart the script
