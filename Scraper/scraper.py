from __future__ import print_function
import argparse
import math
import numpy as np
import urllib.request
import certifi
from bs4 import BeautifulSoup
import requests

checkin_year = 2021
checkout_year = 2021
checkin_month = 9
checkout_month = 9
checkin_monthday = 23
checkout_monthday = 25
num_adults = 2
num_children = 0
num_rooms = 1

for pages in range(1,2):
    addition = ""
    #if pages > 1:
    #    offset = (pages-1)*25
    #    addition = "&amp;offset=" + str(offset)

    url = "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1DCAEoggI46AdIM1gEaFCIAQGYAQm4A" \
        "RnIAQ_YAQPoAQGIAgGoAgO4Aoqo9YcGwAIB0gIkNWMxN2ZmNWEtZWY0My00YjEyLThlMWUtOGRjNmYwMjY0N2Fk2AIE4AIB&" \
        "sid=eda1e1c59139636c0ad160f126898051&sb=1&sb_lp=1&src=index&src_elem=sb&error_url=https%3A%2F%2F" \
        "www.booking.com%2Findex.en-gb.html%3Flabel%3Dgen173nr-1DCAEoggI46AdIM1gEaFCIAQGYAQm4ARnIAQ_YAQPoA" \
        "QGIAgGoAgO4Aoqo9YcGwAIB0gIkNWMxN2ZmNWEtZWY0My00YjEyLThlMWUtOGRjNmYwMjY0N2Fk2AIE4AIB%3Bsid%3Deda1e" \
        "1c59139636c0ad160f126898051%3Bsb_price_type%3Dtotal%26%3B&ss=Canterbury%2C+Kent%2C+United+Kingdom" \
        "&is_ski_area=&checkin_year="+str(checkin_year)+"&checkin_month="+str(checkin_month)+"&checkin_monthday="+str(checkin_monthday)+"&checkout_year="+str(checkout_year)+"&" \
        "checkout_month="+str(checkout_month)+"&checkout_monthday="+str(checkout_monthday)+"&group_adults="+str(num_adults)+"&group_children="+str(num_children)+"&no_rooms="+str(num_rooms)+"&b_h4u_keep_f" \
        "ilters=&from_sf=1&ss_raw=canter&ac_position=0&ac_langcode=en&ac_click_type=b&dest_id=-2591722" \
        "&dest_type=city&place_id_lat=51.279945&place_id_lon=1.080748&search_pageview_id=92015545418a019" \
        "5&search_selected=true&search_pageview_id=92015545418a0195&ac_suggestion_list_length=5&ac_sugges" \
        "tion_theme_list_length=0"

    #url = "https://www.booking.com/searchresults.en-gb.html?label=gen173nr-1FCAEoggI46AdIM1gEaFCIA" \
    #      "QGYAQm4ARnIAQ_YAQHoAQH4AQuIAgGoAgO4AueU64cGwAIB0gIkNTU3YWM2NmUtNzcwOC00ZDU1LTliYTktNDBk" \
    #      "MWE3ZTEzZTVl2AIG4AIB&sid=eda1e1c59139636c0ad160f126898051&sb=1&sb_lp=1&src=index&src_elem" \
    #      "=sb&error_url=https%3A%2F%2Fwww.booking.com%2Findex.en-gb.html%3Flabel%3Dgen173nr-1FCAEogg" \
    #      "I46AdIM1gEaFCIAQGYAQm4ARnIAQ_YAQHoAQH4AQuIAgGoAgO4AueU64cGwAIB0gIkNTU3YWM2NmUtNzcwOC00ZDU1" \
    #      "LTliYTktNDBkMWE3ZTEzZTVl2AIG4AIB%3Bsid%3Deda1e1c59139636c0ad160f126898051%3Bsb_price_typ" \
    #      "e%3Dtotal%26%3B&ss=Edinburgh%2C+United+Kingdom&is_ski_area=&checkin_year="+str(checkin_year)+"&" \
    #      "checkin_month=" + str(checkin_month) + "&checkin_monthday="+str(checkin_monthday)+"&checkout_year="+str(checkout_year)+"&checkout_month="+str(checkout_month)+"&checkout_monthday="+str(checkout_monthday)+"&group_adults="+str(num_adults)+"&group_children="+str(num_children)+"&no_rooms="+str(num_rooms)+"&b_h4u_keep_filters=&from_sf=1&dest_id=-2595386&dest_type=city&search_pageview_id=231161f3baf800b5&search_selected=true"
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

    response = requests.get(url, headers=headers)
    #print(response.url)
    soup = BeautifulSoup(response.content, 'lxml')

    #test_string = '.sr_property_block'
    #test_string = 'div.sr_item'
    #test_string = 'div'
    #test_string = 'sr_property_block'
    #mydivs = soup.find_all("div", {"class_": "sr_item"})
    #print(mydivs)
    #out = soup.select("div.sr_item")
    #print(out)
    #for a in soup.find_all('a', href=True):
    #    print("Url found: ", a['href'])

    iter = 0
    breaker = False
    hotel_names = soup.select('div.sr-hotel__title-wrap')
    prices = soup.select('div.roomPrice')  ###this may be the fake high price
    display_prices = soup.select('div.bui-price-display__value')
    for i in range(len(hotel_names)):
        hotel_name = hotel_names[i].get_text()
        names = hotel_name.split("\n")
        a = prices[i].get_text()
        a = a.split("\n")
        display_price = display_prices[i].get_text().split("\n")
        print(display_price[1], names[4], a[8])

#link = soup.select("a.sr_pagination_link")
link = soup.select("a.bui-pagination__link")
for i in range(1,len(link)):
    new_page_url = "https://www.booking.com/" + link[i]['href']
    response = requests.get(new_page_url, headers=headers)
    soup = BeautifulSoup(response.content, 'lxml')
    hotel_names = soup.select('div.sr-hotel__title-wrap')
    prices = soup.select('div.roomPrice')  ###this may be the fake high price
    display_prices = soup.select('div.bui-price-display__value')
    for i in range(len(hotel_names)):
        hotel_name = hotel_names[i].get_text()
        names = hotel_name.split("\n")
        price = prices[i].get_text()
        price = price.split("\n")
        display_price = display_prices[i].get_text().split("\n")
        print(display_price[1], names[4], a[8])
#for a in soup.find_all('a.bui-pagination__link'):
#    len(a)
#    print(a.get_text())
#print(len(link))
#print(link[0].get_text())
#print(link[1].get_text())
#print(link[2].get_text())

"""
for item in soup.select(test_string):
        print("Item Found")
        try:
            content = item.select('div.sr_item_content') #[0].get_text()
            for i in range(len(content)):
                #test = content.select('span.sr-hotel__name')
                #print(test[0].get_text())
                block_row = content[i].select('div.sr_property_block_main_row')
                #print(block_row.get_text())
                #print(len(block_row))
                #this = block_row[0].select('div.sr_item_main_review')
                item_main_block = block_row[0].select('div.sr_item_main_block')
                #print(item_main_block[0].get_text())
                title_wrap = item_main_block[0].select('div.sr-hotel__title-wrap')
                #print(title_wrap[0].get_text())
                #title = title_wrap[0].select('h3.sr-hotel__title ')
                #print(title.get_text())
                #hotel_title_class = title[0].select('a.js-sr-hotel-link hotel_name_link url')
                #print(hotel_title_class[0].get_text())
                #hotel_title = hotel_title_class[0].select('span.sr-hotel__name')
                #print(hotel_title[0].get_text())
            #print("----- ------ ------")
            #breaker = True
            #break
        except Exception as e:
            print("Didn't work")
        iter += 1
        if breaker:
            break
            break"""


def links(url):
    try:
        # Open URL for reading the HTML (allowing SSL certificates with certifi)
        fp = urllib.request.urlopen(url, cafile=certifi.where())

        # Read data as a bytearray
        data = fp.read()

        # Convert into a utf8 string
        string = data.decode("utf8")

        # Close URL
        fp.close()

        # Initialize beautiful soup in order to scrape the web
        soup = BeautifulSoup(string, features='html.parser')

        # Obtain all href for each link
        links = map(lambda a: a.get('href'), soup.find_all('a'))

        # Filter the ones that begin with http
        result = filter(lambda link: link.startswith('http'), links)

        # Return it as a list
        return list(result)
    except:
        return []











