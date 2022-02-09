from __future__ import print_function
import argparse
import math
import numpy as np
import urllib.request
import certifi
from bs4 import BeautifulSoup
import requests



url = "https://smarkets.com/sport/football"

#headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}
#headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv;90.0) Gecko/20100101 Firefox/90.0)'}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, 'lxml')
#print(soup)

iter = 0
item_tile = soup.select('div.event-info-container')
teams = soup.select("div.teams")
print(teams)
#item_tile = soup.find_all('div.event-info-container')
#item_tile = soup.select('a.title with-score')
#item_tile = soup.select('div.team ')
print(type(item_tile))
print(item_tile)
print(len(item_tile))
print(item_tile[5])
#print(item_tile[0].get_text())
















