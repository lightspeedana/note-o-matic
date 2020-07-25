import requests
from bs4 import BeautifulSoup
from typing import Union

def parse_webpage(url: str) -> Union[str, str]:
    """
        Get the content of a webpage or article to be able to look through

        str url: Url of the page to parse
        
        Return: title of text, text 
    """
    r = requests.get(url)
    if r.status_code != 200:
<<<<<<< HEAD
	    return None
=======
        return None
>>>>>>> d3ac74b5f2dbb395c9053d47bda669fe2577ec52
    soup = BeautifulSoup(r.content, 'html.parser')
    paragraphs = [x.get_text() for x in soup.find_all('p')]
    title = soup.find('h1').get_text()
    return title, ' '.join(paragraphs)
