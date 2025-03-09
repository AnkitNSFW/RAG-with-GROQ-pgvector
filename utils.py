import requests
from bs4 import BeautifulSoup

def scrape_wiki_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find the main content element (usually <div id="mw-content-text"> for wikipedia)
    content_div = soup.find(id='mw-content-text')
    
    # Find all paragraphs within the main content
    paragraphs = content_div.find_all('p')

    # Dropind all the thing with calss "reference", because they contain citation text with [0-9] and don't contribute to the content.
    for para in paragraphs:
        for ref in para.find_all('a', class_='reference'):
            ref.decompose()

        for ref in para.find_all('sup', class_='reference'):
            ref.decompose()
    
    text = '\n'.join([p.get_text() for p in paragraphs])
    text = text.replace('\n', ' ')
    return text

if __name__ == "__main__":
    url = 'https://en.wikipedia.org/wiki/Spider-Man'
    content = scrape_wiki_content(url)
    print(len(content))
    print(content)
