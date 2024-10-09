from setuptools import setup, find_packages

print('Found packages:', find_packages())
setup(
    description='TAPE as a package',
    name='tape',
    packages=find_packages()
)
