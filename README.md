# Fair Privacy

This repository contains the code developed for the ICML submission on the topic of fairness and privacy in machine learning. 


## Requirements

To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## Main File
- "alooa.py": Conduct an approximate leave-one-out attack on auditing datasets to obtain privacy audit results across different various ML training algorithms.
- "looa.py": Conduct a leave-one-out attack on a data record to obtain privacy audit results across different various ML training algorithms.
- "compare.py": Compares the performance of ALOOA and LOOA, including per-record and group-level. It also analyzes the Group Privacy Risk and Group Privacy Risk Parity metrics across multiple attacks, as well as the impact of different models and hyperparameter settings on Group Privacy Risk Parity in ALOOA.
- "repeat_exp.py": Independently runs ML training algorithms multiple times to assess utility.
- "measure.ipynb": Measures Individual Privacy Risk, Group Privacy Risk, and Group Privacy Risk Parity based on the audit results from attacks, and model accuracy based on the results of 'repeat_exp.py'.