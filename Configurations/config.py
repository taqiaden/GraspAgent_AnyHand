import logging
import os
import re
import sys

import numpy as np
import smbclient
import torch

where_am_i = os.popen('hostname').read()
where_am_i = re.sub(r"[\n\t\s]*", "", where_am_i)

use_xyz= True

home_dir = '/home/taqiaden/'
if where_am_i=='yumi':
    home_dir = '/home/yumi/'

solution_name='GraspAgent_AnyHand'
ip_address=r'\\10.5.12.167'

untested_model_stamp= 'untested'

check_points_extension='.pth.tar'

if where_am_i=='chaoyun-server': # server
    check_points_directory=ip_address+r'/taqiaden_hub/NSL_model_state/'


    def configure_smbclient():
        # initialize smbclient
        smbclient.ClientConfig(username='taqiaden', password='774631499')
        # to hide INFO logging messages of smbclient
        logging.disable(sys.maxsize) # hide_smbclient_log
    configure_smbclient()

elif where_am_i=='yumi': #edge unit
    check_points_directory=ip_address+r'/taqiaden_hub/NSL_model_state/'

elif where_am_i=='yons-MS-7D99':
    check_points_directory=r'/home/yons/code/GraspAgent/check_points/'
elif where_am_i=='taqiaden':
    check_points_directory=r'/media/taqiaden/42c447a4-49c0-4d74-9b1f-4b4b5cbe7486/taqiaden_hub/NSL_model_state/'
else:
    r'./check_points/'


counter=0
while os.path.split(os.getcwd())[-1]!=solution_name:
    os.chdir('../')
    counter+=1
    assert counter<100, f'{counter}'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


