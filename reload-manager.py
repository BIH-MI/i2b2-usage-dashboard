import sys
import subprocess
import time
import signal

def run_dash_app():
    """
    Function to run the Dash app as a subprocess.
    """
    return subprocess.Popen(
        [sys.executable, 'i2b2-usage-dashboard.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

def main():
    RESTART_INTERVAL = 30  # Restart interval in seconds

    while True:
        print("Starting Dash app...")
        process = run_dash_app()
        print(f"Dash app started with PID {process.pid}")

        try:
            # Wait for the specified interval before restarting
            time.sleep(RESTART_INTERVAL)
            print("## Restarting Dash app...")

            # Terminate the Dash app process
            process.terminate()
            try:
                # Wait up to 10 seconds for the process to terminate gracefully
                process.wait(timeout=10)
                print("Dash app terminated gracefully.")
            except subprocess.TimeoutExpired:
                # If the process doesn't terminate, kill it forcefully
                print("Dash app did not terminate in time. Killing...")
                process.kill()
                process.wait()
                print("Dash app killed.")

        except KeyboardInterrupt:
            print("Shutting down Dash app manager.")
            process.terminate()
            try:
                process.wait(timeout=5)
                print("Dash app terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("Dash app did not terminate in time. Killing...")
                process.kill()
                process.wait()
                print("Dash app killed.")
            sys.exit(0)

        except Exception as e:
            print(f"An error occurred: {e}")
            process.terminate()
            process.wait()

if __name__ == "__main__":
    main()
