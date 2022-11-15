from selenium import webdriver
from bs4 import BeautifulSoup as bs
import urllib.request
import os
from tqdm import tqdm
import time
keyword = input("검색어 입력:")
url = 'https://www.google.com/search?q=' + keyword
url = url+'&sxsrf=ALiCzsaDuFT_3KHkvRKVAa9GTXHBmubliQ:1662005683847&source=lnms&tbm=isch&sa=X&ved=2ahUKEwih1ZnT3fL5AhWnrlYBHcfxA0QQ_AUoAXoECAEQAw&biw=528&bih=546&dpr=1.65'

driver = webdriver.Chrome('C:\py_data/chromedriver.exe')
driver.get(url)
time.sleep(2)

for i in range(10):
    driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
    time.sleep(2)
    
    if i == 5:
        driver.find_element_by_css_selector('#islmp > div > div > div > div.gBPM8 > div.qvfT1 > div.YstHxe > input').click()
        
html_source = driver.page_source
soup=bs(html_source,'html.parser')

img_list = soup.find('div',class_='islrc').find_all('img')

fDir = 'c:/py_data/'
fName=os.listdir(fDir)

fName_dir=keyword+'0'
cnt=0

while True:
    if fName_dir not in fName:
        os.makedirs(fDir+fName_dir)
        break
    cnt+=1
    fName_dir=keyword+str(cnt)
    
print(fDir+fName_dir,'로 폴더 생성')

cnt=0
for img_url in tqdm(img_list,desc = '저장중...'):
    try:
        urllib.request.urlretrieve(img_url['src'],'c:/py_data/'+fName_dir+'/'+keyword+str(cnt)+'.jpg')
    except:
        urllib.request.urlretrieve(img_url['data-src'],'c:/py_data/'+fName_dir+'/'+keyword+str(cnt)+'.jpg')
    cnt +=1
driver.close()