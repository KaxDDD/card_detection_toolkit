
import time
from pyclashbot.bot.card_detection import capture_card_images
from pyclashbot.emulators import Memu
from pyclashbot.utils.logger import Logger

def main():
    """
    Main function to run the card image capture script.
    """
    logger = Logger()
    logger.change_status("Starting card image capture script...")
    
    # Initialize the emulator
    try:
        emulator = Memu(logger)
    except Exception as e:
        logger.change_status(f"Error initializing emulator: {e}")
        return

    logger.change_status("Emulator initialized.")
    logger.change_status("Please open Clash Royale to the main screen or a battle.")
    logger.change_status("The script will start capturing in 5 seconds...")
    time.sleep(5)
    
    # Run the capture function
    capture_card_images(emulator, logger)
    
    logger.change_status("Card image capture finished.")
    logger.change_status("You can now find the images in the 'card_capture_output' directory.")
    logger.change_status("Please label them with the correct card name (e.g., 'knight.png').")

if __name__ == "__main__":
    main()
