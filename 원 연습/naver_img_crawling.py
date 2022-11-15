from selenium import webdriver
from bs4 import BeautifulSoup as bs
import urllib.request
import os
from tqdm import tqdm
import time

keyword = input("검색어 입력:")
url = 'https://search.naver.com/search.naver?where=image&sm=tab_jum&query=' + keyword
driver = webdriver.Chrome('C:\py_data/chromedriver.exe')
driver.get(url)
time.sleep(2)

for i in range(7):
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(2)

html_source = driver.page_source
soup=bs(html_source,'html.parser')
time.sleep(2)

img_list = soup.find('div',class_='photo_tile _grid').find_all('img', class_='_image _listImage')
img_list = set(img_list)

fDir = 'c:/py_data/'
fName=os.listdir(fDir)

fName_dir='네이버'+keyword+'0'
cnt=0

while True:
    if fName_dir not in fName:
        os.makedirs(fDir+fName_dir)
        break
    cnt+=1
    fName_dir='네이버'+keyword+str(cnt)
    

cft=0
for img_url in tqdm(img_list,desc = '저장중...'):
    main_dir = 'c:/py_data/'+fName_dir+'/'+keyword+str(cft)+'.jpg'
    urllib.request.urlretrieve(img_url['src'],main_dir)
    cft += 1

    