# lab-project-fairness

For more details look at [Project.pdf](https://github.cs.huji.ac.il/jjgold/lab-project-fairness/blob/master/Project.pdf) and [here](https://arxiv.org/pdf/1707.00044.pdf)

## Requirements
Run:
```
pip3 install -r requirements.txt
```

## old_project
In order to run the old project run:
```
python3 old_project/fairness.py
```

## fairness_project
In order to run this project with the ```compas``` dataset run:
```
python3 fairness_project/start.py fairness_project/options/compas.json
```

The ```default``` dataset can be run in a similar way:
```
python3 fairness_project/start.py fairness_project/options/default.json
```

In order to rub the project with synthetic data run:
```
python3 fairness_project/start.py <epsilon>
```

### Results
The outputs results can be seen in the ```results``` folder.

** The ```default``` data set result are with train set size of 20% because the dataset is quite large and it ran for a long time.


1. compas dataset with both weights
2. adult dataset with both weights
3. default dataset with both weights
4. compas dataset with one weights
5. adult dataset with one weights
6. default dataset with one weights
