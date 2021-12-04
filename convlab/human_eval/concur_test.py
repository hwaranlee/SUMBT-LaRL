import requests

input = 'hi I want to book a hotel'
requests.post('http://localhost:10004', json={'input': input, 'agent_state': {}})