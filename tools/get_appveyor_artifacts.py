# -*- coding: utf-8 -*-
"""
Download AppVeyor artifacts for shapely

See https://www.appveyor.com/docs/api/projects-builds/

Created on Tue Jan 21 12:06:02 2020

@author: Mike Taves
"""

import os
import requests
import textwrap

account = 'frsci'
slug = 'shapely'

api = 'https://ci.appveyor.com/api'


def download_job_artifacts(job_id):
    resp = requests.get(f'{api}/buildjobs/{job_id}/artifacts')
    for artifact in resp.json():
        filename = artifact['fileName']
        print(f'downloading: {filename}')
        resp = requests.get(f'{api}/buildjobs/{job_id}/artifacts/{filename}')
        save_path = os.path.split(filename)[-1]
        with open(save_path, 'wb') as f:
            f.write(resp.content)


def main(branch='master'):
    print(f"Gathering last build information for branch '{branch}'")
    resp = requests.get(f'{api}/projects/{account}/{slug}/branch/{branch}')
    data = resp.json()
    if 'build' not in data:
        raise ValueError(data)
    build = data['build']
    jobs = build.get('jobs', [])
    if not jobs:
        raise ValueError('no jobs found')
    print('Found {0} jobs for last build for {1}'.format(len(jobs), branch))
    print('Commit id: ' + str(build['commitId']))
    print(textwrap.indent(build['message'], '> '))
    # Check if all jobs were built successfully with one artifact
    job_ids = []
    for job in jobs:
        if job.get('status') != 'success':
            raise ValueError('not all jobs were built successfully')
        elif job.get('artifactsCount', 0) == 0:
            raise ValueError('not all jobs produced artifacts')
        job_ids.append(job['jobId'])
    for job_id in job_ids:
        download_job_artifacts(job_id)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--branch', default='master',
        help='branch to download artifacts from (default: %(default)s)')
    args = parser.parse_args()
    main(**vars(args))
