import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import trafilatura
import time

def get_driver(link):
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-popup-blocking")

    prefs = {
        "profile.default_content_setting_values.popups": 0,
        "profile.default_content_settings.popups": 0,
        "profile.managed_default_content_settings.popups": 0,
    }
    options.add_experimental_option("prefs", prefs)

    driver = uc.Chrome(options=options)
    wait = WebDriverWait(driver, 15)
    driver.get(link)
    return driver, wait

def simple_try(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.text
    except requests.RequestException as e:
        print(f"Request failed: {e}")
    return None

def driver_try(url):
    driver, wait = get_driver(url)
    try:
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        time.sleep(0.1)
        page_source = driver.page_source
        driver.quit()
        return page_source
    except Exception as e:
        print(f"Driver failed: {e}")

    driver.quit()
    return None

def AutoScraper(url):
    html = simple_try(url)

    if not html:
        html = driver_try(url)
    
    if html:
        try:
            resultado = trafilatura.extract(html,include_comments=False, include_tables=False)
            return resultado
        except:
            return None

    print('ERROR OBTENIENENDO EL TEXTO')
    return None