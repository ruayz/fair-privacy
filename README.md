# Fair Privacy

This repository contains the code developed for the NIPS submission on the topic of fairness and privacy in machine learning. 


## Requirement

To install the dependencies, run the following command:
```bash
conda env create -f env.yaml
```

## Main File
- "main.py": Conduct attacks on auditing datasets to obtain privacy audit results.
- "repeat_exp.py": Independently runs ML training algorithms multiple times to assess utility and outcome fairness.
- "measure.ipynb": Measures Individual Privacy Risk, Group Privacy Risk, and Group Privacy Risk Parity based on the audit results from attacks (PA-GA, PA-GBA, and PA-ALOOA), and model accuracy and outcome fairness measurements based on the results of 'repeat_exp.py'.
- "alooa.py": Conduct an approximate leave-one-out attack on auditing datasets(in setting of m=600).
- "looa.py": Conduct a leave-one-out attack on a data record.
- "compare.py": Compares the performance of PA-ALOOA and PA-LOOA, including per-record and group-level.