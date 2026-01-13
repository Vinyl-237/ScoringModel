from pydantic import BaseModel

class ClientData(BaseModel):
    bureau_DAYS_CREDIT_mean: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    REGION_RATING_CLIENT: int
    NAME_INCOME_TYPE_Working: int
    DAYS_LAST_PHONE_CHANGE: int
    CODE_GENDER_M: int
    DAYS_ID_PUBLISH: int
    pos_MONTHS_BALANCE_min: float
    EXT_SOURCE_1: float
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float