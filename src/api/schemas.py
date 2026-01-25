from pydantic import BaseModel

class ClientData(BaseModel):
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    bureau_DAYS_CREDIT_UPDATE_mean: float
    REGION_RATING_CLIENT_W_CITY: int
    NAME_INCOME_TYPE_Working: int
    DAYS_LAST_PHONE_CHANGE: int
    DAYS_ID_PUBLISH: int
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float