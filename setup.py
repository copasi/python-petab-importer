from setuptools import setup
import versioneer

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()
    
# Replace relative image path with absolute GitHub URL
long_description = long_description.replace(
    './doc/demo.gif',
    'https://github.com/copasi/python-petab-importer/raw/master/doc/demo.gif'
)


ENTRY_POINTS = {
    'console_scripts': [
        'copasi_petab_import=copasi_petab_importer.convert_petab:main',
        'copasi_petab_gui=copasi_petab_importer.PEtab:petab_gui',
    ]
}

setup(
    name='copasi-petab-importer',
    packages=['copasi_petab_importer'],
    package_dir={'copasi_petab_importer': 'copasi_petab_importer'},
    url='https://github.com/copasi/python-petab-importer',
    license='Artistic-2.0',
    author='Frank T. Bergmann',
    author_email='frank.bergmann@bioquant.uni-heidelberg.de',
    description='COPASI PEtab Importer',
    entry_points=ENTRY_POINTS,
    install_requires=['numpy', 'pandas', 'python-copasi', 'python-libsbml', 'pyyaml'],
    extras_require={'gui': ['PyQt5']},
    package_data={'copasi_petab_importer': ['*.ui']},
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    long_description=long_description,
    long_description_content_type='text/markdown',
)
