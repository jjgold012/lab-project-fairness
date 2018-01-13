# lab-project-fairness

For more details look at [Project.pdf](https://github.cs.huji.ac.il/jjgold/lab-project-fairness/blob/master/Project.pdf)

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

### Results
The outputs results can be seen in the ```results``` folder. 

** The ```default``` data set result are with train set size of 20% because the dataset is quite large and it ran for a long time. 