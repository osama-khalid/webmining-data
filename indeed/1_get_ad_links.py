import csv
from indeed import IndeedClient

client = IndeedClient(publisher = "4405390899698537")

QUERY = 'java'		#Query
LOCATION = '52246'	#Location
LIMIT = 500		#Approximate number of enteries 


indeed_data = []
isVisit = []
i=0
while len(isVisit) < LIMIT:
	i=i+1
	params = {
	    'q' : QUERY,
	    'l' : LOCATION,
	    'start':str(i*10),
	    'userip' : "1.2.3.4",
	    'useragent' : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_2)"
	}

	search_response = client.search(**params)
	for item in search_response['results']:
		jobkey = item['jobkey']
		joburl = item['url']
		jobcity = item['city']
		jobstate = item['state']
		jobcountry = item['country']
		jobdate = item['date']
		if jobkey not in isVisit:
			indeed_data.append([jobkey,joburl,jobcity,jobstate,jobcountry,jobdate])
			isVisit.append(joburl)
			isVisit = list(set(isVisit))
	
with open('ad_links.csv', mode='w') as indeed_file:
    indeed_writer = csv.writer(indeed_file, delimiter=',')
    indeed_writer.writerows(indeed_data)