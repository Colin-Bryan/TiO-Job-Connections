import requests

def skills_API(skill):
    '''
    Queries the API Layer Skills database for the 'skill' argument and returns response

        Parameters:
            skill (str): One or two words to search in the Skills database

        Returns:
            result (str): JSON result from the API query
    '''
    # Use API Layer's skills database
    url = "https://api.apilayer.com/skills?q={}".format(skill)

    # Pass in API key to headers
    headers= {
    "apikey": "# Hidden as this has been commented out"
    }

    # Get response and result
    response = requests.request('GET', url, headers=headers)
    result = response.json()
    
    # If response is successful, return result
    if response.status_code == 200:
        return result

    # Else, get the exception
    raise Exception(result.get('message'))
