# End to End Student Performance Analysis 

----------------------------------

setup.py
--------
* It is responsible in  creating my machine learning application as a package.
* ie, building our application as a package itself.
---------------------------------
__init__.py
-------------
* whichever the folder contains the __init__.py , it consider as a package.


-----------------------------------------

It is not feasiable to give the required packages in install_requires = ['pandas','numpy','seaborn',..]

so instead , 
install_requires = get_requirements('requirements.txt') in setup.py 


-----------------------------------------

-e. automaticaly triggers  setup.py


While we give the command pip install -r requirements.txt , since there is -e . in requirements.txt it maps to setup.py .It gives an indication that setup.py file is there and automatticaly the package will get build


-----------------------------------------

Hence you can see an mlproject.egg-info in which the package and author information 


-----------------------------------------

src is the folder in which the entire proejct components is to be in . 

create a folder components in the folder src and a file __init__.py to make it as a package.

components are the file  which includes different stages of projects like data ingestion , data transformation , data validation 


-----------------------------------------

1. create a file data_ingestion.py 
Data ingestion is the process of importing, collecting, or importing data from various sources into a storage or processing system.




2. Create a data_transformation.py 
Data transformation is a crucial step in data processing pipelines and is often necessary to prepare data for analysis, visualization, or other downstream tasks.

3. Create a model_trainer.py file 
This file is an essential component of the machine learning workflow and is used to define, train, and evaluate models based on given datasets. 
Mainly for the training purpose 


These components are nothing but the modules we are going to use in the particular project  



-----------------------------------------

Then we are going to create the pipelines 

src--> pipelines--> train_pipeline.py 
src--> pipelines--> predict_pipelines.py 
and create __init__.py to make it as package


-----------------------------------------

Then in src  we are going to create the logger.py file ,exception.py and utils.py 


-----------------------------------------
Gonna write our own ecxception

The sys module in Python is a built-in module that provides access to system-specific parameters and functions. It can be used to get information about the Python interpreter, the operating system, and the environment in which the Python script is running. It can also be used to manipulate different parts of the Python runtime environment.


-----------------------------------------

Performed All the necessary Data cleaning , Feature Engineering and Data visualization part EDA.ipynb file 



-----------------------------------------
Then proceeded to Model_training.ipynb in which we are gonna train the model 


-----------------------------------------

While we are encoding the categorical features , since there are very less number of features we can use **one hot encoding** . If there are more number of features we can use  **target guided ordinal encoding**


-----------------------------------------

we are then going to use the pipeline for all the process instead of doing it separately 

ColumnTransformer is a preprocessing class provided by scikit-learn, a popular machine learning library in Python. It allows you to apply different transformations to different columns of your dataset, enabling you to construct a single transformer that handles multiple preprocessing steps simultaneously.



-----------------------------------------


Once we completed the part in jupyter notebook we are going to proceed to modular coding. 



-----------------------------------------


**Data Ingestion**

The main aim is to read the dataset from specific data source. And split the data into train and test 


" from dataclasses import dataclass "
The dataclass decorator from the dataclasses module in Python provides a way to automatically generate special methods for classes. It's particularly useful for creating classes that primarily store data and don't have much behavior. 



all the outputs will automatically stored in the artifacts folder 

    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','raw.csv')
From the above steps data ingestion components now know where to save the train path , test path  and raw path . 


-----------------------------------------

If you are not defining varibales you can use @dataclass . But if you are defining some functions inside the class , use the constructor part .  


def initiate_data_ingestion(self): 
In the function defined above , 
if you stored the data in some databases , you can connect and read it from here . for eg: MongoDB



test the data ingestion file and artifacts folder is being created 


-----------------------------------------

now include .artfacts in .gitignore so that its not get saved .



-----------------------------------------

Now moving  to **DataTransformation**

The main aim of Data Transformation part is to do data cleaning , data encoding and feature scaling , feature engineering , dimensionality reduction, data imputation. 


-----------------------------------------

* from sklearn.compose import ColumnTransformer
**ColumnTransformer** is used to apply different preprocessing transformations to different columns or subsets of columns within a dataset. It helps in organizing and applying various transformations in a structured manner, especially when dealing with mixed data types (e.g., numerical and categorical) in machine learning pipelines.

* from sklearn.impute import SimpleImputer
**SimpleImputer** is used to handle missing values (NaNs) in the dataset by replacing them with a specified value (e.g., mean, median, or most frequent value). It's a simple and commonly used technique for data preprocessing to ensure that the dataset is ready for analysis or modeling.


* from sklearn.pipeline import Pipeline
**Pipeline** is used to sequentially apply a series of transformations or processing steps to the data. It's particularly useful for organizing complex workflows, such as preprocessing data (e.g., imputation, scaling) and fitting models (e.g., classifiers, regressors) in a single coherent pipeline. This helps in encapsulating and automating the entire machine learning workflow.


* from sklearn.preprocessing import OneHotEncoder,StandardScaler


**OneHotEncoder** is used to encode categorical variables into a numerical format, particularly when the categorical variables have no ordinal relationship (i.e., they're nominal). It creates binary dummy variables for each category, making it suitable for use with machine learning algorithms.


**StandardScaler** is used for feature scaling, which ensures that numerical features are on the same scale (i.e., have zero mean and unit variance). It's important for many machine learning algorithms, particularly those sensitive to the scale of input features, such as gradient descent-based algorithms.


-----------------------------------------


-----------------------------------------


-----------------------------------------



-----------------------------------------

-----------------------------------------

-----------------------------------------

-----------------------------------------

-----------------------------------------

-----------------------------------------