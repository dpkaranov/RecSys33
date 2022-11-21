#import libraries
import os
import uvicorn
from fastapi.openapi.utils import get_openapi
from service.api.app import create_app
from service.settings import get_config

#import variables
config = get_config()
app = create_app(config)

#add info into swagger
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Team33 RecSys Service",
        version="0.0.1",
        description="This is a custom openapi of RecSys project made by team_33.",
        routes=app.routes,
    )
    openapi_schema["paths"]["/reco/{model_name}/{user_id}"]["get"]["responses"] = {"200":{"description":"Successful Response",
                                                                    "content":{"application/json":{"schema":{"$ref":"#/components/schemas/RecoResponse"}}}},
                                                                    "401":{"description":"Error on authorization. Incorrect token.",
                                                                    "content":{"application/json":{"schema":{"$ref":"#/components/schemas/RecoResponse"}}}},
                                                                    "404":{"description":"Not found.",
                                                                    "content":{"application/json":{"schema":{"$ref":"#/components/schemas/RecoResponse"}}}}}
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
