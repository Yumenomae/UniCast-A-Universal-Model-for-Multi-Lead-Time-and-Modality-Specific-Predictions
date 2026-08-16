import h5py
import numpy as np
from typing import Any, Dict, Tuple, Dict, List, Sequence

def get_data_given_path(
    path: str
) -> Dict:
    """Prepare the data given the data path and variables.
    
    """
    with h5py.File(path, 'r') as f:
        data = {
            main_key: {
                sub_key: np.array(value) for sub_key, value in group.items()
        } for main_key, group in f.items() if main_key in ['input']}
    return data

days_in_months = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

class check_h5df_data():

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.data = None
        self.data_hour_in_year = None

    def load_from_path(self, path: str):

        data = get_data_given_path(path)
        self.data = data['input']
        return 
    
    def load_from_year_hour(self, year: int, hour: int):

        if year in [2019, 2020, 2021]: prefix = 'train'
        elif year == 2022: prefix = 'val'
        else: prefix = 'test'

        path = f'/home-ssd/Users/gm_lhy/zhengjq/Ours/data/{prefix}/{year}_{hour}.h5'
        self.load_from_path(path)
        self.data_hour_in_year=hour
    
    def load_from_date(self, year: int, month: int, day: int, hour_in_day: int):

        past_days_from_first_year = days_in_months[:month-1].sum()
        past_days_from_this_month = day - 1

        past_hours = (past_days_from_first_year + past_days_from_this_month) * 24 + hour_in_day

        if year == 2020 and month > 2: past_hours = past_hours + 24

        self.load_from_year_hour(year, past_hours)
    
    def check_date(self):
        
        assert self.data != None
        print('Date:', self.data['time'])
        print('Hour in year:', self.data_hour_in_year)
    

if __name__ == '__main__':
    wrapper = check_h5df_data()
    path = '/home-ssd/Users/gm_lhy/zhengjq/Ours/data/test/2023_1057.h5'
    wrapper.load_from_path(path)
    wrapper.check_date()

