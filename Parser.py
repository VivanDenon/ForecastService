from random import randint
from ssl import SSLError
from typing import final
from typing_extensions import Self
from bs4 import BeautifulSoup
from matplotlib.pyplot import pause
import requests
import pandas as pd
import time
from urllib3 import exceptions as ex

word = 'Санкции в отношении РФ'
url = 'https://ria.ru/search/?query=' 


def getDates():
    csv_path = 'USDRUB_000101_220625.csv'
    df = pd.read_csv(csv_path)
    dates = pd.to_datetime(df.pop('<DATE>'), format='%d/%m/%y')
    return dates.tolist()


def onError(errors):
    for error in errors:
        print(error)


def parse(dates):
    dates = dates[int(len(dates)):]
    data = []
    s = requests.session()
    if s:
        for d in dates:
            url_with_sort = url + word + f'&sort=date&date_from={d.date()}&date_to={d.date()}'
            print(url_with_sort)
            try:
                result = s.get(url_with_sort, verify=False, timeout=(10, 20))
                if result:
                    soup = BeautifulSoup(result.text, "html.parser")
                    sf = soup.find('div', class_='rubric-count m-active')
                    sf = sf.find('span')
                    if sf is not None:
                        data.append(sf.text)
                        print(sf.text)
                    else:
                        counts.append(0)
                else:
                    print(result)
                    counts.append(0)
                result.close()
            except Exception as inst:
                for error in errors:
                    print(error)
                break
            time.sleep(randint(10, 20))
    return data


def writeToFile(data):
    print(data)
    file = open('data.txt', 'w')
    for d in data:
        file.write(d + '\n')
    file.close()


dates = getDates()
print(len(dates))
data = parse(dates)
print(data)
writeToFile([str(dates[i].date()) + ',' + 0 for i in range(int(len(dates) / 4))] + [str(dates[int(len(dates) / 4) + i].date()) + ',' + data[i] for i in range(len(data))])