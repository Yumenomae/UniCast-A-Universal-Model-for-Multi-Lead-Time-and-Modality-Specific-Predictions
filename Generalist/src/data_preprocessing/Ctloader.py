import numpy as np
import datetime as dt
import collections
import re
import os
# from visualization_from_data import visualization_from_numpy

def var_info_generator(fps):
    info = collections.OrderedDict()
    lv = 0
    for idx, i in enumerate(fps):
        if i[1] == '0':
            level = 1
        else:
            level = int(i[1])
        if i[2] == '0':
            scale = 1
        else:
            scale = int(i[2])
        info[i[0]] = {'level': level, 'scale': scale, 'long_name': i[3]}
        info[i[0]]['start_lv'] = lv
        lv = lv + info[i[0]]['level']
    return info

# to figure out ctl date-time information
def parse_time(time_str):
    regex = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)mn)?((?P<seconds>\d+?)s)?')
    parts = regex.match(time_str)
    if not parts:
        return
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return dt.timedelta(**time_params)
# Class for all ctl informations

class Ctl2dict:
    def __init__(self, file_path):
        x_pattern = re.compile(r'([x,X][a-zA-Z]{3})\s+(\d+)\s+([a-zA-Z]{6})\s+(-?\d+\.\d+)\s+(\d+\.\d+)')
        y_pattern = re.compile(r'([y,Y][a-zA-Z]{3})\s+(\d+)\s+([a-zA-Z]{6})\s+(-?\d+\.\d+)\s+(\d+\.\d+)')
        z_pattern = re.compile(r'([z,Z][a-zA-Z]{3})\s+(\d+)\s+[[a-zA-Z]{6}')
        zlv_pattern = re.compile(r'\n\s+(\d+\.\d*)')
        t_pattern = re.compile(r'([t,T][a-zA-Z]{3})\s+(\d+)\s+linear\s+(\w*)\s*(\w*)')
        vnum_pattern = re.compile(r'([v,V][a-zA-Z]{3})\s+(\d+)\n')
        var_pattern = re.compile(r'(\w+)\s+(\d+)\s+(\d+)\s+(\w+.*)')
        with open(file_path, 'r') as f:
            ctlstr = ''.join(f.readlines())
        fps = re.findall(x_pattern, ctlstr)[0]
        self.x_cor = {'x_points': int(fps[1]), 'interval_mode': fps[2], 'start_value': float(fps[3]),
                      'resolution': float(fps[4]), 'long_name': 'longitude'}
        self.x_cor['end_value'] = self.x_cor['start_value'] + (self.x_cor['x_points'] - 1) * self.x_cor['resolution']
        fps = re.findall(y_pattern, ctlstr)[0]
        self.y_cor = {'y_points': int(fps[1]), 'interval_mode': fps[2], 'start_value': float(fps[3]),
                      'resolution': float(fps[4]), 'long_name': 'latitude'}
        self.y_cor['end_value'] = self.y_cor['start_value'] + (self.y_cor['y_points'] - 1) * self.y_cor['resolution']
        fps = re.findall(z_pattern, ctlstr)[0]
        self.z_cor = {'z_points': fps[1], 'long_name': 'z levels hPa'}
        fps = re.findall(zlv_pattern, ctlstr)
        self.z_cor['levels_array'] = np.array(fps, dtype=float)
        fps = re.findall(t_pattern, ctlstr)[0]
        self.info = {'steps': int(fps[1]),  # 'date':dt.datetime.strptime(fps[2],'%Hz%d%b%Y'),
                     'time_perstep': parse_time(fps[3])}
        fps = re.findall(var_pattern, ctlstr)
        self.var_set = var_info_generator(fps)
        self.info['var_count'] = len(self.var_set)
        
# Class for model data information and getdata functions
class ModelDataReader(object):
    def __init__(self, ctl_p=None):
        self.ctl = Ctl2dict(ctl_p)
        self.hold = 1
        self.model_dtype = '>f4'
    def get_var(self, file, var=[], t=0):
        if len(var) == 0:
            var = list(self.ctl.var_set)
        def read_var(iof, v):
            lvs = self.ctl.var_set[v]['start_lv']
            for idx, i in enumerate(range(self.ctl.var_set[v]['level'])):
                iof.seek(((self.ctl.x_cor['x_points'] * self.ctl.y_cor['y_points'] + self.hold * 2) * (
                        lvs + t * self.ctl.info['var_count']) + 1) *
                         int(self.model_dtype[2]))
                v = np.frombuffer(iof.read(self.ctl.x_cor['x_points'] * self.ctl.y_cor['y_points'] *
                                           int(self.model_dtype[2])), dtype=self.model_dtype)
                lvs += 1
                if idx == 0:
                    vs = v.reshape((1, self.ctl.y_cor['y_points'], self.ctl.x_cor['x_points']))
                else:
                    v = v.reshape((1, self.ctl.y_cor['y_points'], self.ctl.x_cor['x_points']))
                    vs = np.concatenate([vs, v])
            return vs
        var_dict = {}
        with open(file, 'rb') as f:
            for v in var:
                var_dict[v] = read_var(f, v)
        return var_dict
    
if __name__ == '__main__':
    
    tmp = ModelDataReader('post.ctl')



