import requests
import shutil


# Replace "dogvcat" with your model name, and replace the link with your desired input image link
payload = {'modelname': 'dogvcat'}
link = "https://c2.staticflickr.com/6/5093/5389312711_08e67fa19b_b.jpg"

def download_file(url):
    local_filename = url.split('/')[-1]
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

    return local_filename


files = {'file': open(download_file(link), 'rb')}
r = requests.post('http://0.0.0.0:5432/classify', data=payload, files= files)
print(r.text)


# The output will be something like this:
# {"class": "Cat", "score": "0.99"}