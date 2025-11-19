from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

def wake_up_streamlit():
    url = "https://briefingengine.streamlit.app/"
    print(f"Visiting {url}...")

    # Setup Headless Chrome
    chrome_options = Options()
    chrome_options.add_argument("--headless") # Runs without a UI
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Initialize the driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        driver.get(url)
        # IMPORTANT: Wait for the app to actually load/wake up
        # Streamlit apps take 10-20 seconds to boot from sleep.
        print("Page loaded. Waiting 30 seconds for app to boot...")
        time.sleep(30) 
        
        title = driver.title
        print(f"Success! Page title is: {title}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        driver.quit()

if __name__ == "__main__":
    wake_up_streamlit()
