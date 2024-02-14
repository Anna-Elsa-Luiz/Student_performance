# End to End Student Performance Analysis 

============================================================================================

setup.py
--------
* It is responsible in  creating my machine learning application as a package.
* ie, building our application as a package itself.
============================================================================================

__init__.py
-------------
* whichever the folder contains the __init__.py , it consider as a package.


===========================================================================================

It is not feasiable to give the required packages in install_requires = ['pandas','numpy','seaborn',..]

so instead , 
install_requires = get_requirements('requirements.txt') in setup.py 

===========================================================================================

-e. automaticaly triggers  setup.py


While we give the command pip install -r requirements.txt , since there is -e . in requirements.txt it maps to setup.py .It gives an indication that setup.py file is there and automatticaly the package will get build

============================================================================================

Hence you can see an mlproject.egg-info in which the package and author information 

============================================================================================

src is the folder in which the entire proejct components is to be in . 

create a folder components in the folder src and a file __init__.py to make it as a package.

components are the file  which includes different stages of projects like data ingestion , data transformation , data validation 

============================================================================================

1. create a file data_ingestion.py 
Data ingestion is the process of importing, collecting, or importing data from various sources into a storage or processing system.




2. Create a data_transformation.py 
Data transformation is a crucial step in data processing pipelines and is often necessary to prepare data for analysis, visualization, or other downstream tasks.

3. Create a model_trainer.py file 
This file is an essential component of the machine learning workflow and is used to define, train, and evaluate models based on given datasets. 
Mainly for the training purpose 


These components are nothing but the modules we are going to use in the particular project  


===========================================================================================

Then we are going to create the pipelines 

src--> pipelines--> train_pipeline.py 
src--> pipelines--> predict_pipelines.py 
and create __init__.py to make it as package

===========================================================================================

Then in src  we are going to create the logger.py file ,exception.py and utils.py 

============================================================================================
Gonna write our own ecxception

The sys module in Python is a built-in module that provides access to system-specific parameters and functions. It can be used to get information about the Python interpreter, the operating system, and the environment in which the Python script is running. It can also be used to manipulate different parts of the Python runtime environment.

============================================================================================

Performed All the necessary Data cleaning , Feature Engineering and Data visualization part EDA.ipynb file 


============================================================================================

Then proceeded to Model_training.ipynb in which we are gonna train the model 

============================================================================================

While we are encoding the categorical features , since there are very less number of features we can use **one hot encoding** . If there are more number of features we can use  **target guided ordinal encoding**

============================================================================================

we are then going to use the pipeline for all the process instead of doing it separately 

ColumnTransformer is a preprocessing class provided by scikit-learn, a popular machine learning library in Python. It allows you to apply different transformations to different columns of your dataset, enabling you to construct a single transformer that handles multiple preprocessing steps simultaneously.


============================================================================================


Once we completed the part in jupyter notebook we are going to proceed to modular coding. 


============================================================================================
