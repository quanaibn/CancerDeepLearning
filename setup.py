from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='CancerDeepLearning',
    url='CancerDeepLearning',
    author='https://github.com/BiomedicalMachineLearning/CancerDeepLearning.git',
<<<<<<< HEAD
    author_email='quan.nguyen@uq.edu.au, a.su@uq.edu.au, x.tan3@uq.net.au',
=======
    author_email='quan.nguyen@uq.edu.au; a.su@uq.net.au',
>>>>>>> 9f1d8a51e03be2838c003479fd59d4870c0410a6
    # package modules
    packages=['CancerDeepLearning'],
    #declare script
    scripts=[bin/]
    # dependencies
    install_requires=['numpy', 'tensorflow'],
    # version
    version='0.1',
    # license
    license='MIT',
    description='Developing a package using genomics and image data for cancer diagnosis',
)

