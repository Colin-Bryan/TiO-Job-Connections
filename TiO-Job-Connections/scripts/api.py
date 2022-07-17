import requests

def skills_API(skill):
    # Use API Layer's skills database
    url = "https://api.apilayer.com/skills?q={}".format(skill)

    # CB 7.16 - Hide API key in a secrets file before sharing?
    headers= {
    "apikey": "MOKg5sAQ8JxpSVFKVsqtvIcUGRDA97aM"
    }

    # Get response and result
    response = requests.request('GET', url, headers=headers)
    result = response.json()
    
    if response.status_code == 200:
        return result

    raise Exception(result.get('message'))
