# Import necessary setup packages
from setuptools import find_packages, setup
from typing import List

# "-e." triggers setup.py from the requirements.txt
setup_trigger = "-e ."


def get_requirements(file_path: str) -> List[str]:
    '''
    Returns a list of required libraries/packages
    '''
    requirements = []

    with open(file_path) as file_obj:
        # Save each line in the requirements list
        requirements = file_obj.readlines()

        # Use list comprehension to remove '/n' character
        requirements = [req.replace("\n", "") for req in requirements]

        if setup_trigger in requirements:
            requirements.remove(setup_trigger)

    return requirements


setup(
    name='FIAfrica',
    version='1.0',
    author='Conyema',
    author_email='onyemachinedum@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
