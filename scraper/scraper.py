from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import requests
import re
import os


class Scraper:
    '''
    It is used to search for a particular text on the web and scrape the top x pages from the search results.
    By default, it scrapes the top 3 pages.
    '''
    num_pages = 3
    DRIVER_PATH = ""
    service = None

    
    def __init__(self, num_pages=3):
        '''
        Constructor for the Scraper class
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is 3.
        '''
        self.num_pages = num_pages
        self.DRIVER_PATH = os.environ.get('SELENIUM_WEB_DRIVER_CHROME_PATH')
        if self.DRIVER_PATH == None:
            raise Exception("Please set the environment variable SELENIUM_WEB_DRIVER_CHROME_PATH")

        self.service = Service(executable_path=self.DRIVER_PATH)
    

    def __parse_web_search_results(self, content, article_list=[]):
        '''
        This function is used to parse the google search results, and stores the article url and text in form of a list.
        @param content: The content of the page to be parsed
        @param article_list: The list to store the article url and text
        '''
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "style", "header", "footer"]):
            script.decompose()
        body = soup.find('body')
        articles = body.find_all('div', class_='kvH3mc')
        for article in articles:
            if(article.find('h3') == None or article.find('a') == None):
                continue
            obj = {
                'url': article.find('a')['href'],
                'title': article.find('h3').text,
            }
            article_list.append(obj)

    
    def __search_text_on_web(self, query, num_pages=num_pages):
        '''
        This function is used to search for the given text on the web and store the search results in a list.
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is passed from the constructor.
        '''
        browser = webdriver.Chrome(service=self.service)
        browser.implicitly_wait(0.5)
        browser.get("https://www.google.com")
        
        #identify search box
        search = browser.find_element(By.XPATH, "//input[@name='q']")
        
        #enter search text
        search.send_keys(query)
        time.sleep(0.2)
        
        #perform Google search with Keys.ENTER
        search.send_keys(Keys.ENTER)

        # extract the search results and store it in a list
        article_list = []

        # for next page id is 'pnnext'
        hasContent = True
        pageNo = 1
        
        # fetch the content of top 3 pages
        while(pageNo<=num_pages and hasContent):
            content = browser.page_source
            self.__parse_web_search_results(content, article_list)
            try:
                next_page = browser.find_element(By.ID, "pnnext")
                next_page.click()
                time.sleep(0.5)
                pageNo += 1
            except:
                hasContent = False
        
        browser.quit()
        return article_list
    
    
    def __scrape_content_from_url(self, url):
        '''
        This function is used to scrape the content from the given url. 
        It removes all the unnecessary tags and returns the text.
        @param url: The url from which the content is to be scraped. 
        '''
        # get the content from the url
        content = requests.get(url).content
        # parse the content using beautiful soup
        soup = BeautifulSoup(content, 'html.parser')
        body = soup.find('body')

        if body == None:
            return ""
        
        # Remove the header and footer
        if body.header:
            body.header.decompose()
        if body.footer:
            body.footer.decompose()
        
        # Remove the media tags
        media_tags = ['video', 'audio', 'picture', 'source', 'img']
        for tag in media_tags:
            for media in body.find_all(tag):
                media.decompose()
        
        # Remove all inputs
        input_tags = ['input', 'form', 'select', 'textarea']
        for tag in input_tags:
            for input in body.find_all(tag):
                input.decompose()

        # Remove the scripts
        for script in body.find_all('script'):
            script.decompose()
        
        # Remove all the buttons
        for button in body.find_all('button'):
            button.decompose()

        # Remove navigation
        for nav in body.find_all('nav'):
            nav.decompose()
        
        # Remove all icons
        for i in body.find_all('i'):
            i.decompose()
        
        # Remove all links
        for link in body.find_all('a'):
            link.decompose()
        
        # Remove all code tags
        code_tags = ['code', 'kbd', 'samp', 'var']
        for tag in code_tags:
            for code in body.find_all(tag):
                code.decompose() 
        
        # Remove unnecessary tags
        unnecessary_tags = ['s', 'del']
        for tag in unnecessary_tags:
            for t in body.find_all(tag):
                t.decompose()
            
        text = body.text
        
        # remove the newlines
        text = text.replace('\n', ' ')
        # remove the extra spaces
        text = re.sub(' +', ' ', text)
        return text
    

    def scrape(self, query, num_pages=num_pages):
        '''
        This function is used to scrape the web for the given query and returns the web articles in the form of a list.
        @param query: The text to be searched on the web
        @param num_pages: The maximum number of pages to be scraped from the search results. Default is passed from the constructor.
        '''
        # search for the given query on the web
        article_list = self.__search_text_on_web(query, num_pages)
        # scrape the content from the url
        for article in article_list:
            article['text'] = self.__scrape_content_from_url(article['url'])
        return article_list

# sc = Scraper()
# result = sc.scrape("Agri Ministry Captures India's Initiative 'International Year of Millets' In Tableau")
# print(result)