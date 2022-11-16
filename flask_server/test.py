import pip._vendor.requests 

response = pip._vendor.requests.post('http://localhost:8000/add/misspelled', json={"predicted":"plud", "autocorrected":"plus"}, verify=False)
print(response.text)

response = pip._vendor.requests.post('http://localhost:8000/add/word', json={"word":"hello"}, verify=False)
print(response.text)

response = pip._vendor.requests.post('http://localhost:8000/add/word', json={"word":"was"}, verify=False)
print(response.text)