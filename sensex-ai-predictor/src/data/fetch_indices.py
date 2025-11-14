import requests
from bs4 import BeautifulSoup

def fetch_sensex_indices():
    """
    Fetches the latest Sensex indices and their real market values from a financial website.
    Returns a dictionary with stock symbols as keys and their corresponding values.
    """
    url = "https://www.example.com/sensex"  # Replace with the actual URL
    response = requests.get(url)

    if response.status_code != 200:
        print("Error fetching data from the website.")
        return {}

    soup = BeautifulSoup(response.content, 'html.parser')
    indices = {}

    # Assuming the indices are in a table format, adjust the selectors as needed
    table = soup.find('table', {'class': 'sensex-table'})  # Replace with actual class
    rows = table.find_all('tr')[1:]  # Skip header row

    for row in rows:
        columns = row.find_all('td')
        if len(columns) >= 2:
            symbol = columns[0].text.strip()
            value = float(columns[1].text.strip().replace(',', ''))  # Convert to float
            indices[symbol] = value

    return indices