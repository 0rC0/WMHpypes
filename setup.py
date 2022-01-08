from distutils.core import setup
exec(open('wmhpypes/version.py').read())
setup(
    name='wmhpypes',
    version='__version__',
    packages=['wmhpypes', 'wmhpypes.utils', 'wmhpypes.interfaces', 'wmhpypes.workflows'],
    url='https://github.com/0rC0/WMHpypes',
    license='BSD 3-Clause License',
    author='Andrea Dell Orco',
author_email = 'andrea.dellorco@dzne.de',
               description = 'Deeplearning white matter hyperintensities segmentation'
)
