class Config:
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            self.__setattr__(k, v)