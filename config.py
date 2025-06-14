from datetime import datetime

# For Discourse fetching, filtering, and processing data
DATE_FORMAT = "%Y-%m-%d"
START_DATE = datetime.strptime("2025-01-01", DATE_FORMAT)
END_DATE = datetime.strptime("2025-06-13", DATE_FORMAT)
# Number of discourse pages to scrape
PAGES = 10
PYTESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



# For course content fetching, filtering, and processing data
PLAYWRIGHT_BROWSERS_PATH = r"E:\TDS\virtual-ta-tds\venv\Lib\site-packages"
