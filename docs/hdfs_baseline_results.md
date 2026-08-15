# HDFS anomaly-detection baseline

Dataset: LogHub HDFS v1  
Subset: 1,000 Normal + 1,000 Anomaly blocks  
Model: TF-IDF (unigrams + bigrams) + Logistic Regression  
Train/test split: 75/25 stratified  
Selected threshold: 0.4  

## Results

| Metric | Score |
|---|---:|
| Precision | 0.815 |
| Recall | 0.932 |
| F1 | 0.869 |
| PR-AUC | 0.958 |

## Decision

Threshold 0.4 was selected because it retained near-best F1 while detecting substantially more anomalies than threshold 0.5.