import pip._vendor.requests 

# response = pip._vendor.requests.post('http://localhost:8000/add/misspelled', json={"predicted":"plua", "autocorrected":"plus"}, verify=False)
# print(response.text)

# response = pip._vendor.requests.post('http://localhost:8000/add/word', json={"word":"hi"}, verify=False)
# print(response.text)

response = pip._vendor.requests.post('http://localhost:8000/add/word', json={"word":"want"}, verify=False)
print(response.text)