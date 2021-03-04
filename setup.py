from setuptools import setup

ENTRY_POINTS = {
    'console_scripts': [
        'copasi_petab_import =convert_petab:main',
    ]
}

setup(
    name='python-petab-importer',
    version='0.0.1',
    packages=[''],
    url='https://github.com/copasi/python-petab-importer',
    license='Artistic-2.0',
    author='Frank T. Bergmann',
    author_email='frank.bergmann@bioquant.uni-heidelberg.de',
    description='COPASI PEtab Importer',
    entry_points=ENTRY_POINTS,
)
