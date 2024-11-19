import os
import time
import csv
from pynput import keyboard

# Initialize key_data to store keylogs
key_data =  []

def on_press(key):
    """Capture key press event and store the key and timestamp"""
    try:
        key_data.append((str(key.char), time.time(),  'press'))
    except AttributeError:
        # Handle special keys (SHift, Enter, etc.)
        key_data.append((str(key), time.time(), 'press'))
        
def on_release(key):
    """Capture key release event and store the key and timestamp."""
    try:
        key_data.append((str(key.char), time.time(), 'release'))
    except AttributeError:
        key_data.append((str(key), time.time(), 'release'))

    # Stop the keylogger when the ESC key is pressed
    if key == keyboard.Key.esc:
        # Stop listener
        return False
        

def save_key_data(filename):
    """Save the keylog data to a CSV file inside a specific folder."""
    
    # Define the folder path inside your project (e.g., 'keylogs' folder)
    folder_path = os.path.join(os.getcwd(), 'keylogs')
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Full path of the file to be saved
    file_path = os.path.join(folder_path, filename)
    
    # Write the keylog data to the CSV file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Timestamp', 'Action'])  # Header
        writer.writerows(key_data)  # Keylog data

    print(f"Data saved at: {file_path}")

def keylog_user(user_id):
    """Run the keylogger for a specific user and save the data."""
    print(f"Start typing for User {user_id}. Press ESC to stop.")

    # Clear the previous key data
    global key_data
    key_data = []

    # Start capturing keylogs
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()  # Keeps the script running until manually stopped by ESC key
        
    # Save the data in the 'keylogs' folder
    save_key_data(f"user_{user_id}_keylogs.csv")
    print(f"Data saved for User {user_id} in 'keylogs/user_{user_id}_keylogs.csv'.")

if __name__ == "__main__":
    num_users = int(input("Enter the number of users to log: "))
    
    for  user in range(1, num_users + 1):
        input(f"Press Enter to start keylogging for User {user}...")
        keylog_user(user) # Track each user's typing behaviour
        print( f"User {user} data collection complete.\n")