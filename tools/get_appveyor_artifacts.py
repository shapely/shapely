# -*- coding: utf-8 -*-
"""
Download AppVeyor artifacts for shapely

See https://www.appveyor.com/docs/api/projects-builds/

Created on Tue Jan 21 12:06:02 2020

@author: Mike Taves
"""

import os
import requests

account = 'frsci'
slug = 'shapely'
branch = 'master'

api = 'https://ci.appveyor.com/api'

resp = requests.get(f'{api}/projects/{account}/{slug}/branch/{branch}')

build = resp.json()['build']
jobs = build['jobs']

print('Getting {0} jobs for last build for {1} with commit id {2}'
      .format(len(jobs), branch, build['commitId'][:8]))

for job in jobs:
    job_id = job['jobId']
    resp = requests.get(f'{api}/buildjobs/{job_id}/artifacts')
    for artifact in resp.json():
        filename = artifact['fileName']
        print(f'downloading: {filename}')
        resp = requests.get(f'{api}/buildjobs/{job_id}/artifacts/{filename}')
        save_path = os.path.split(filename)[-1]
        with open(save_path, 'wb') as f:
            f.write(resp.content)
