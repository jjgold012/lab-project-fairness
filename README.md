# lab-project-fairness

For more details look at [Project.pdf](https://github.cs.huji.ac.il/jjgold/lab-project-fairness/blob/master/Project.pdf) and [here](https://arxiv.org/pdf/1707.00044.pdf)

## Requirements
Run:
```
pip install -r requirements.txt
```

## old_project
In order to run the old project run:
```
python old_project/fairness.py
```

## fairness_project
In order to run this project with on one of the data sets in the folder ```fairness_project/datasets/``` choose an option
file from ```fairness_project/options/``` and Run
```
python fairness_project/start.py fairness_project/options/<any options json file>
```

Or you can create your own option file. Here's an example:
```
{
  "description":      "compas dataset with protected value race",   #Description of the problem
  "data_set":         "compas-dataset",                             #The data set to use
  "file":             "compas-scores-two-years.csv",                #The csv file name
  "data_headers":     "age,sex,race,priors_count,c_charge_degree",  #Which headers to use from the data
  "protected":        "race",                                       #The protected header in the data set
  "tag":              "is_recid",                                   #The target header
  "fp":               true,                                         #Take FPR diff into account?
  "fn":               false,                                        #Take FNR diff into account?
  "objective_weight": 1,                                            #The objective: Accuracy + (if fp==true)objective_weight*FPR_diff + (if fn==true)objective_weight*FNR_diff
  "weight":           {"gt": 0, "lt": 400},                         #The relaxed objective weight range
  "weight_res":       21,                                           #The number of weights to test in the range 
  "gamma":            {"gt": 0, "lt": 10},                          #The regularization parameter (gamma) range
  "gamma_res":        11,                                           #The number of gamma to test in the range
  "train_size":       0.50,                                         #The train size precentage from the entire data
  "num_of_tries":     5,                                            #The number of times to check each weight,gamma combination

  "filters": [                              #Filters to apply on the data set
    {
      "header": "c_charge_degree",          #Filter if the header is on of the values
      "values": ["0"]
    },
    {
      "header": "days_b_screening_arrest",
      "values": [""],
      "range": {                            #Filter if the header is numeric and is either larger then 30 or smaller then -30
        "type": "or",
        "gt": 30,
        "lt": -30
      }
    }
  ],

  "headers_integer_values": [               #Replace sting values with integers
    {
      "header": "race",                     #Replace for the header: 'race', if 'Caucasian' replace with 0, if 'African-American' replace with 1 
      "Caucasian": 0,
      "African-American": 1
    },
    {
      "header": "c_charge_degree",
      "M": 0,
      "F": 1
    }
  ],
  "headers_operation": []
}

```

** _Notice that the options ```json``` file can't contain any comments! because it will not parsed correctly if it does._

In order to rub the project with synthetic data run:
```
python fairness_project/start.py <epsilon>
```

The synthetic data is generated like this
1. Choose target value ```y``` randomly from ```{0,1}```.
2. Choose ```x_0 = y``` with probability ```1 - epsilon``` else ```x_0 = 1 - y```
3. Choose ```x_1 = y``` with probability ```1 - 2*epsilon``` else ```x_0 = 1 - y```
4. Run the problem with protected value ```x_0```

### Results
The outputs results can be seen in the ```results``` folder.

** _The ```default, adult, ucla-law-school``` data sets result are with train set size of 20% because the data set is quite large and it ran for a long time._


