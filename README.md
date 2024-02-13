End to End Student Performance Analysis 

===========================

setup.py
--------
* It is responsible in  creating my machine learning application as a package.
* ie, building our application as a package itself.
===============================

__init__.py
-------------
* whichever the folder contains the __init__.py , it consider as a package.


=====================================

It is not feasiable to give the required packages in install_requires = ['pandas','numpy','seaborn',..]

so instead , 
install_requires = get_requirements('requirements.txt') in setup.py 

=======================================

-e. automaticaly triggers  setup.py


While we give the command pip install -r requirements.txt , since there is -e . in requirements.txt it maps to setup.py .It gives an indication that setup.py file is there and automatticaly the package will get build

============================================

Hence you can see an mlproject.egg-info in which the package and author information 

=========================================