from distutils.core import setup

setup(
    name='Git_Repos_Tracker',
    version='0.1.0',
    packages=['git_repos_tracker', 'git_repos_tracker.demo',
              'git_repos_tracker.util'],
    license='MIT',
    install_requires=[
        'git-python',
    ],
    description=''
)
