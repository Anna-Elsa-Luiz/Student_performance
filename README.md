
<h1>Student Exam Performance Prediction</h1>

<p>This project aims to solve the problem of predicting the outcome of students in final exams using supervised machine learning techniques from Sklearn. It's a regression problem where predictions are made based on a dataset of classroom students' biodata along with their reading and writing scores to predict the maths score of a particular student. Various regression techniques, including XGBoost and Random Forests of decision trees, have been studied and implemented in the project.</p>

<p>For detailed exploratory data analysis (EDA) and feature engineering, check out the notebook directory.</p>

<p>Performance of different regression models, such as Linear Regression, Lasso Regression, Ridge Regression, K-Neighbors Regressor, Decision Tree, Random Forest Regressor, XGBRegressor, CatBoosting Regressor, and AdaBoost Regressor, were compared to determine the best-performing models for our dataset. These models were then utilized to predict the maths score of a particular student based on user input from the Flask application.</p>

<p>The dataset used in this project is sourced from Kaggle and stored in GitHub as well as inside the notebook directory. The features in the dataset include:</p>

<ul>
  <li>Gender: Student's gender ('Male', 'Female')</li>
  <li>Race/Ethnicity: Ethnicity of students (Group A, B, C, D, E)</li>
  <li>Parental Level of Education: Parents' final education (Bachelor's degree, Some college, Master's degree, Associate's degree, High school)</li>
  <li>Lunch: Whether the student had lunch before the test (Standard or Free/Reduced)</li>
  <li>Test Preparation Course: Whether the student completed the test preparation course before the test</li>
  <li>Math Score (integer)</li>
  <li>Reading Score (integer)</li>
  <li>Writing Score (integer)</li>
</ul>

<h2>Installing</h2>

<p>Environment setup:</p>
<code>conda create --prefix venv python==3.9 -y</code><br>
<code>conda activate venv/</code><br>

<p>Install requirements and setup:</p>
<code>pip install -r requirements.txt</code><br>

<p>Run application:</p>
<code>python app.py</code>

<h2>Built with</h2>

<ul>
  <li>Flask</li>
  <li>Python 3.9</li>
  <li>Machine learning</li>
  <li>Scikit-learn</li>
</ul>

<h2>Industrial Use Cases</h2>

<p>Models Used:</p>

<ul>
  <li>Linear Regression</li>
  <li>Lasso Regression</li>
  <li>Ridge Regression</li>
  <li>K-Neighbors Regressor</li>
  <li>Decision Tree</li>
  <li>Random Forest Regressor</li>
  <li>XGBRegressor</li>
  <li>CatBoosting Regressor</li>
  <li>AdaBoost Regressor</li>
</ul>

<p>After hyperparameter optimization, the top two models selected were XGBRegressor and Random Forest Regressors, which were then used in the pipeline. GridSearchCV was used for hyperparameter optimization in the pipeline.</p>



<ul>
  <li><strong>Artifact</strong>: Stores all artifacts created from running the application</li>
  <li><strong>Components</strong>: Contains all components of the machine learning project, including 
   DataIngestion, DataValidation, DataTransformations, ModelTrainer, ModelEvaluation</li>
</ul>

<p>Custom logger and exceptions are used in the project for better debugging purposes.</p>






### Steps I followed and information reagrding some important files  

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
In DataTransformation also we are going to do the exception handling.

-----------------------------------------
we have constructed a dataingestion_config class in Dataingestion.py  we must constrct a  similar config in datatransformation.py also 

-----------------------------------------

def get_data_transformer_object(self):
* It is to create all my pickle files which will be responsible for  converting categorical to numerical , or if I want to perform   Standardscaler and all.  

-----------------------------------------

Then provided the numerical features =[] and categorical features =[]


-----------------------------------------
An object is being created from Pipeline()
Numerical pipline is initialized 

for handling missing values as well standardizing the numerical features we have applied the numerical pipeline 

-----------------------------------------


numerical_pipeline = Pipeline(
                 steps=[
                     ('imputer',SimpleImputer(strategy ='median')),
                     ('scaler',StandardScaler())
                       ]
                                           )




-----------------------------------------


categorical_pipeline = Pipeline(
                 
                 steps= [
                     ('imputer',SimpleImputer(strategy='most_frequent')),
                     ('one_hot_encoder',OneHotEncoder()),
                     ('scaler',StandardScaler(with_mean=False))
                         ]
                                             )



-----------------------------------------
**Numerical and Categorical Pipelines Explained:**

In machine learning and data science, we often work with datasets that contain a mix of numerical features (represented by numbers) and categorical features (represented by non-numerical values like text, dates, or categories). To prepare these features for effective use in machine learning models, we need to apply different preprocessing techniques, depending on the data type. This is where numerical and categorical pipelines come in.

### Numerical Pipelines:

* Purpose: Handle numerical features by applying specific transformations that are suitable for their continuous nature.



Common Steps:
* Imputation: Fill in missing values using strategies like mean/median imputation or more advanced techniques.
* Scaling: Rescale numerical features to a common range (e.g., min-max scaling, standardization) to prevent features with larger scales from dominating the model.
* Feature Engineering: Create new features by combining existing numerical features (e.g., ratios, differences, products).
* Dimensionality Reduction: If high dimensionality is an issue, techniques like Principal Component Analysis (PCA) can reduce the number of features.


#### Categorical Pipelines:

Purpose: Prepare categorical features for use in models, as they cannot be directly processed by most algorithms.


Common Steps:
* Encoding: Convert categorical values into numerical representations suitable for the model. Common techniques include one-hot encoding, label encoding, and target encoding.
* Handling Ordinal Data: If categories have a natural order (e.g., clothing sizes), encode them in a way that preserves this order (e.g., integer or frequency encoding).
* Dimensionality Reduction: If there are many unique categories, consider techniques like feature hashing or embedding to reduce the number of dimensions.
-----------------------------------------

preprocessor = ColumnTransformer(

              ("num_pipeline",numerical_pipeline,numerical_columns),
              ("cat_pipeline",categorical_pipeline,categorical_columns)  
                                            )
**(pipeline name , what pipeline it is , coloumn type)**

-------------------------------------------
Then defined a function 

def initiate_data_transformation(self,train_path,test_path):Th

for reading the train and test dataset ,numerical and target column


#### Steps:

1. Read Data:

- Reads training and testing data from CSV files using pd.read_csv.
- Logs informational messages using logging.info.


2. Get Preprocessing Object:

- Calls self.get_data_transformer_object() to obtain a preprocessing object (which is likely created elsewhere and encapsulates preprocessing steps).

3. Identify Columns:

- Defines target_column_name (the column to predict, e.g., "math_score").
- Defines num_columns (list of numerical feature columns, e.g., ["writing_score", "reading_score"]).

4. Separate Features and Target:

- Creates input_feature_train_df by dropping the target column from the training data.
- Creates target_feature_train_df by selecting the target column from the training data.
- Similarly, creates input_feature_test_df and target_feature_test_df for the testing data.

5. Apply Preprocessing:

- Logs a message indicating preprocessing application.
- Uses preprocessing_obj.fit_transform on the training input features to fit the preprocessor and transform the data.
- Uses preprocessing_obj.transform on the testing input features to transform the data without refitting (since it was fitted on the training data).

6. Combine Features and Target:

- Concatenates the transformed input features and the target column (as NumPy arrays) into train_arr and test_arr for training and testing, respectively.

7. Save Preprocessing Object (Optional):

- Logs a message about saving the preprocessing object.
- Uses save_object (likely a custom function or library) to save the preprocessing_obj to a file specified by self.data_transformation_config.preprocessor_obj_file_path.
------------------------------------------


save_object function will be in the utils.py 

where utils.py is the file in which all the common functionalities

now in the utils file define save_object 
add dill in the requirements.txt too

with the help of save_object in the utils.py  file we are going to save the pickle file in the hard disk


----------------------------------------


* from src.utils import save_object
Add the above import statement in data_transformation.py


* from src.components.data_transformation import DataTransformation
* from src.components.data_transformation import DataTransformationConfig
Add the above import statement in data_ingestion.py


------------------------------------------

Now we are going to train the model in the model_trainer.py 

we are using the model: Linear regression , DecisionTree Regression , Kneighbors Regressor  and ensemble techniques like Adaboost , catboost , Xgboost , Gradient boosting 



------------------------------------------
after importing the required libraries , modules 

created a dataclass for modeltrainerconfig  for saving the pkl file 
this basicallly gives whatever input I required with model training 

then created another class Modeltrainer 



-------------------------------------------

Then in the class defined a function , def  

initiate_model_trainer(self,train_array,test_array,preprocessor_path):
          
          try:

            logging.info('Splitting training and test input data')
              X_train , y_train , X_test , y_test = (
                   train_array[:,:,-1],
                   train_array[:,-1],
                   test_array[:,:,-1],
                   test_array[:,-1]
                                                    )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            } 

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,x_test= X_test,y_test=y_test , models=models)

train test splitting has been done in the model trainer and the models were defined. 

--------------------------------------------

Then in the utils file create the  evaluate_model()  as follows;


def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report ={}  

        for i in range(len(list(models))):

            model = list(model.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score =r2_score(y_test,y_test_pred)

            report[list(model.keys())[i]]=test_model_score

        return report    

    except Exception as e:
        raise CustomException(e, sys)



-------------------------------------------

Add import statement of r2_score in utils.py 
and import  save_object and evaluate_models in model_trainer.py

-----------------------------------------------

In  model_trainer , found out the best model and given a condition with r2_score

saved the file path in model_trainer.py 

Dumped the best model 

finally , predicted output r2_score and best model

-----------------------------------------------

Then in dataingestion import the modelTrainerconfig and ModelTrainer.  

Initializes the data transformation in dataingestion.py 
train_arr and test_arr are returned from the datatransformation , going to save them in dataingestion


Then initialise the modeltrainer in dataingestion.py 

And finally print the r2_score as the output 
------------------------------------------------

Since we havenot done the hyperparameter tuning , going to do them. 

using 
para=param[list(models.keys())[i]]

gs = GridSearchCV(model,para,cv=3)
gs.fit(X_train,y_train)

model.set_params(**gs.best_params_)
model.fit(X_train,y_train)

in utils.py


And defined the best hyperparameters of each of them  in model_trainer.py 

 params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }




----------------------------------

Now we are going to create the prediction pipeline

Going to create a simple web application which will be interacting with any input we are goin to give.
----------------------------------

created a app.py file 


create a folder templates in which create two files index.html and home.html 
-------------------------------------

Then in the prediction pipleine 

imported the necessary library 
created a predictpipleine class and then the customdata which we given the app.py 

this customdata class will be responsible in mapping all the inputs that given in the html to the backend with the particular values 

-------------------------------------------

we are defining the input variables to be collected from the user to predict the output 

then defined a function to conert the input varibales into dataframe.

-------------------------------------------------

Then defined a funt predict in class predictpipleine:
and saved the model.pkl file and the preprocessor which is responsible for feature scaling , handling categorical features,...

load_object is used to import the pkl file and load it. 


Then in utils.py file , 
define the function :

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

It just opening the file path in read mode and loading the pkl file using dill.

---------------------------------



Then once we load the data , we are goin to scale the data 

and defined it in a try except block 

---------------------------


then in app.py file 
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

-------------------------

In the app.py define the custom data to accept the input form using the user Interface

data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )


And stored it in the dataframe 
pred_df = data.get_data_as_data_frame()      

--------------------------------------


initialize the predictpipeline in app.py 

And then call the predict function , as soon as we call the funt-->go to the predict fucntion  and the transformation, scaling will be happen and prediction


Then store the result in  the results variable and return the output  in the UI 

return render_template('home.html',results=results[0])
results[0]--> in the list format

----------------------------------

Now we need to read the value in  home.html



---------------------------------------

Now we are goin to deploy the project in AWS Elastic Beanstalk

we need to do some configuration before we deploy it in elastic beasnstalk

so we created  folder .ebextension  , in that create a python.config file 

It is used to tell the elastic beanstalk instance about what is the entry point of the application  

just for the sake of deployment create a application.py file and copy the contents in app.py 

make sure to remove the debug=True while deploying 


-----------------------------

now we are going to deploy in aws 

Using Elastic Beanstalk : server or cloud environment of some instance  (linux machine)

The only thing that elasticbeanstalk requires is the configuration we gave in python.config in.ebextensions 


In the github repository you have the code and should be able to go to the elasticbeanstalk.
Inorder to do that we have codepipeline :

Codepipeline basically commit or deploy it in the aws automaticallly as soon as we click a button inside the elastic beanstalk which is like a linux machine. 

so that whenever any changes in the github repo, the codepipeline will automattically commmit it to the elasticbeanstalk


It is a  continous delivery pipeline 

1. first we will create a elastic beanstalk instance which can be a linux machine and create a enviroment and do the setup 

2. we are going to create a code pipeline 
Through this codepipeline will integrate with the github repository and we will continue the depoyment to the elastic beanstalk 



 


<<<<<<< HEAD

=======
>>>>>>> 981c5442641e0e48501a479681be903c51e52c11

Steps

1. ElasticBeanstalk 

2. Create an application

3. application name :Student_Performance

4. platform:python

5. No need to change the default values 

6. create application

7. can skip the steps in btw and submit 

Now we are goin to create the  code pipeline 

8. codepipeline

9. create pipline 

10. pipeline name: studentperformance 

11. no need to chnage the default or advanced settings 

12. source: github version 1

13. connect to github: authoeize : confirm 

14. You have successfully configured the action with the provider.

15. select the repo name: Anna-Elsa-Luiz/Student_performance

16. branch: master
GitHub webhooks (recommended)

17. skip the build stage 
18. deploy provider: AWS Elastic Beanstalk
19. region: US East Virginia 
20. appplication name : Student_Performance_Aws
21. env name: StudentPerformanceAws-env
22. create pipeline 


























--------------------------------------------


Azure deployment


Converting my web application into Docker image 

Once it is converted into Docker image , it is going to deployed in the serveer we are going to use . 

Docker image which is a private image , willl deployed into container registry.

once we upload the docker image , web app will be pulled over in Azure 

-------------------------------

AZURE DEPLOYMENT 

1. Conatiner Registry
2. Docker Setup in local and push container registry 
3. Azure Web app with container 
4. Configured the github Deployment Center 


Azure--> Container registries --> new container registry ---> free trail ---> Resouce group: Create new: testdocker-->registry name: testdockeranna: this registry name will be holding your entire docker images. 

next>>> 

create 

Created a container registry where all the docker images will be conatined or put it here 

Go to resource 

access keys ---> enable Admin user 

user name : testdockeranna
password1 : 57+lj0Gm2bHqc0I7P6va8VQiJfQxyLGoOEkeWXsORe+ACRBiAH99

password2: dz6l8xydbI/je/FgAJISmhb+5iw2fJAKllCLgfNZ4y+ACRBgKab3


create new resource--> web app --> ceate --> 
resouce group name : testdocker 
name: testdockeranna 

publish: Docker Containers 

next>> dockers:
Single container 
Azure container repository
