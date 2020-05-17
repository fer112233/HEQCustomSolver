from distutils.core import setup

setup(name='HEQCustomSolver',
      version='1.0',
      description='Resolvedor de ecuaciones de calor customizable.',
      author='Fernando Pascual',
      author_email='fepaso@edem.es',
      install_requires=['numpy', 'matplotlib', 'matplotlib', 'scipy', 'pygame', 'pygame'],
      console=['HEQ.py']
     )