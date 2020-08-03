# LendingClub Charged-off Loan Prediction
This project aims to predict whether a loan will be charged-off based on historical data from 2016-2019 obtained from LendingClub’s website and its API. 

## Data
  - Historical loan data: https://www.lendingclub.com/info/download-data.action
  - New loan data: https://www.lendingclub.com/developers/api-overview

In addtion, to support feature engineering efforts, macro-economic data of various granularity were sourced from different sources. Those datasets inclue state unemployment rate, population by zip, and income by zip throughout time.

## Methodology
### Handling imbalanced data
Since roughly only 25% of the loans are charged-off vs fully-paid, I explored two ways of handeling the imbalance:

* First approach: Balance the two class by undersampling the fully-paid loans and use ROC-AUC to evaluate the model performance
* Second approach: Do not balance the class and use PR-AUC to evaluate the model performance 

### Model
In this project, I applied LightGBM and used Bayesian Optimization to tune hyperparameters

## Result

|  | precision | recall | f1-score | support |
| ------ | ------ | ------ | ------ | ------ |
| 0.0 | 0.89 | 0.82 | 0.85 | 50,035 |
| 1.0 | 0.32 | 0.47 | 0.38 | 9,148 |
| accuracy |  |  | 0.76 | 59,183 |
| macro avg | 0.61 | 0.64 | 0.62 | 59,183 |
| weighted avg | 0.80 | 0.76 | 0.78 | 59,183 |

The top 5 features of the highest feature importance are:
* intrate: interest rate on the loan
* emptitle_chargeoff_pct: avg. historical charged-off rate per emptitle
* dti: debt-to-income ratio
* annualinc: annual income
* installment: the monthly payment owed by the borrower if the loan originates

The two approach of handeling the imbalanced data render similar important features. 
The second approach gives higher weighted avg. F1-score (0.78) than the first one (0.69).

Excluding other types of loan status (e.g. current, grace-period, late), the model predicts that 18% of the loans would be charged off and the 82% of the loan would be fully-paid among the new loans. 

