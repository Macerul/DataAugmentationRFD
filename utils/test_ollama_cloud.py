from ollama import Client
API_KEY = "ebf71e32976545afbe0d4717de605145.i5B60cJtkWTRCiyjOj4kUPu_" #mary1
API_KEY = "d745d1d30809440b805d7df1a5f91357.AcMeIW8ez4TlapoHdsDVMAIA"
API_KEY = "d745d1d30809440b805d7df1a5f91357.AcMeIW8ez4TlapoHdsDVMAIA"
API_KEY = "c7cd567cc4ff42fdb1a6fde284ad4689.BtdYWB-ulklKgnlu47UWSdN7"
API_KEY   = "78682b51a2f64d4e8e3ba6b73d6e2d11.lP7Jpi4wvKPzX6WlLWxL3Scz"
API_KEY   = "f6df339f761742d3a5110fb7b336c5bc.skzGCAfTLwS11j5o2GoVX4sW" #valeria 2

ollama_client = Client(host="https://ollama.com", headers={"Authorization": "Bearer " + API_KEY})

models = ["gpt-oss:20b-cloud", "gpt-oss:120b-cloud", "devstral-small-2:24b-cloud", "devstral-small-2:24b", "ministral-3:8b", "ministral-3:14b-cloud"]
models = ["gemma3:12b-cloud","gemma4:31b"]
prompt = "Say hello!"

for mdl in models:
    try:
        response = ollama_client.chat(
        model=mdl,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": 0.7},
        )
        print("mdl", response)
    except Exception as e:
        print("[ERROR] mdl: ", mdl, e)