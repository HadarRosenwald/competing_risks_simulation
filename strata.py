from enum import Enum


class Strata(Enum):
    AS = 0
    P = 1
    H = 2
    D = 3

    @classmethod
    def has_value(cls, value):
        return any(value == item.value for item in cls)

    @classmethod
    def safe_get_name(cls, value):
        try:
            return cls(value).name
        except ValueError:
            return "NA"


def get_strata(d0: int, d1: int) -> Strata:
    if d0 == 0:
        if d1 == 0:
            return Strata.AS
        else:
            return Strata.H
    else:
        if d1 == 0:
            return Strata.P
        else:
            return Strata.D
