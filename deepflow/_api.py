import os
import uuid

global api_active
api_active =  0

def promoid(use,key=None):
    if use == "dev":
        key = None
        api_active =  1
        pass
    else:
        if key in os.environ['_df-api-keys']:
            api_active =  0
            pass
        else:
            print("deepflow._api.promoid - invalid api key.")

def new_key():
    nkey = uuid.uuid4().hex
    os.environ['_df-api-keys'] = nkey
