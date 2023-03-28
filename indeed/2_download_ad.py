import csv
import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
options = Options()
#options.binary_location=r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'
driver = webdriver.Chrome(options=options, executable_path='./chromedriver81.exe')
ad_file = 'ad_links.csv'
urlSet = []
i=0
with open(ad_file) as csv_file:   
	csv_reader = csv.reader(csv_file, delimiter=',')
	for row in csv_reader:
		url = row[1]
		if url not in urlSet:
			i+=1
			urlSet.append(url)
			title = row[0]+str(i)
			driver.get(url)
			time.sleep(5)
			path = 'ads/'
			if not os.path.isdir(path):
	   			os.makedirs(path)

			with open(path+title+'.html', 'w+') as f:
			    f.write(driver.page_source)	