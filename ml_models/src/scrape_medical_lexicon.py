"""scrape_medical_terminology.py.
   Created on 25 April 2020.

   Acquire medical terms for use in filtering the claims data.
"""

import string
import requests
from bs4 import BeautifulSoup


def scrape_nhsconditions():
    """Scrape medical terms from NHS conditions page."""
    url = 'https://www.nhs.uk/conditions/#'
    response = requests.get(url, headers=None)
    tp_soup = BeautifulSoup(response.content, 'html.parser')

    return [str(li.get_text().strip()) + '\n' for li in
            tp_soup.find_all('li', {'class': 'nhsuk-list-panel__item'})
            ]


def scrape_everydayhealth():
    """Scrape medical terms from Everyday Health conditions page."""
    url = 'https://www.everydayhealth.com/conditions/'
    response = requests.get(url, headers=None)
    tp_soup = BeautifulSoup(response.content, 'html.parser')

    return [str(li.get_text().strip()) + '\n' for li in
            tp_soup.find_all(
                'li', {'class': 'topicslist__topiccolumncont-item'})
            ]


def scrape_medicalencyclopedia():
    """Scrape terms from MedlinePlus' medical encylcopedia."""
    ency_terms = []
    medencyclopedia_urls = [
        'https://medlineplus.gov/ency/encyclopedia_{}.htm'.format(letter)
        for letter in string.ascii_uppercase
        ]

    for url in medencyclopedia_urls:
        response = requests.get(url, headers=None)
        tp_soup = BeautifulSoup(response.content, 'html.parser')
        ency_terms += [str(li.get_text().strip()) + '\n' for li
                       in tp_soup.find('ul', {'id': 'index'}).find_all('li')]
    return ency_terms


def scrape_tlapcare():
    """Scrape hospital terms from TLAP Care and Support Jargon Buster."""
    content = ''.join(
        open('../../data/misc/TLAP Care and Support Jargon Buster.htm'
            ).readlines())
    tp_soup = BeautifulSoup(content, 'html.parser')

    return [str(li.get_text().strip()) + '\n' for li in
            tp_soup.find_all('a', {'class': 'term'})]


def scrape_nationalcareershealthcare():
    """"Scrape health care careers terms."""
    response = requests.get(
        'https://nationalcareers.service.gov.uk/job-categories/healthcare',
        headers=None
        )
    tp_soup = BeautifulSoup(response.content, 'html.parser')

    return [str(li.get_text().strip()) + '\n' for li in
            tp_soup.find_all('a', {'class': 'dfc-code-search-jpTitle'})]


def scrape_mayoclinic_terms():
    urls = []
    symptoms = []
    letters = string.ascii_uppercase + '0'
    symptoms_urls = ['https://www.mayoclinic.org/symptoms/index?letter={}'.format(
                     letter) for letter in letters]

    diseases_urls = ['https://www.mayoclinic.org/diseases-conditions/index?letter={}'.format(
                     letter) for letter in letters]

    procedures_urls = ['https://www.mayoclinic.org/tests-procedures/index?letter={}'.format(
                     letter) for letter in letters]

    drugs_urls = ['https://www.mayoclinic.org/drugs-supplements/index?letter={}'.format(
                     letter) for letter in letters]
    urls = symptoms_urls + diseases_urls + procedures_urls + drugs_urls

    for url in urls:
        response = requests.get(url, headers=None)
        tp_soup = BeautifulSoup(response.content, 'html.parser')

        if tp_soup.find('div', {'id': 'index'}) is None:
            continue
        if tp_soup.find('div', {'id': 'index'}).find('ol').get_text().strip() == '':
            continue 

        symptoms_links = [li.a['href'] for li in 
        tp_soup.find('div', {'id': 'index'}).find('ol').find_all('li')]

        symptoms += [url.split('/')[2].replace('-', ' ')+'\n' for url in symptoms_links]
    
    print(symptoms)
    print(len(symptoms))
    symptoms = sorted(list(set(symptoms)))
    with open('../../data/misc/terms_mayo_clinic.txt', 'w+') as output_file:
        output_file.writelines(symptoms)




def scrape_all():
    """Scrape all terms."""
    terms = scrape_nhsconditions()
    terms += scrape_everydayhealth()
    terms += scrape_medicalencyclopedia()
    terms += scrape_tlapcare()
    terms += scrape_nationalcareershealthcare()
    

    terms_unique = {}

    for elem in terms:

        if elem in terms_unique:
            continue

        terms_unique[elem] = 1

        if elem.find("(") > -1:
            term = elem
            terms.remove(elem)
            first = term[:term.find("(")] + term[term.rfind(")")+1:]
            second = term[term.find("(")+1: term.rfind(")")]

            terms.append(first.strip() + '\n')
            terms.append(second.strip() + '\n')

    with open('../../data/medical_event_claims/terms.txt', 'w+') as output_file:
        output_file.writelines(terms)


if __name__ == '__main__':
    TERMS = scrape_all()
