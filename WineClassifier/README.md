# Wine Classifier

Goal: Create a model to classify a variety of wine into 3 categories: Low, Med, High.

### Features:
><b>Fixed Acidity</b>:          The amount of fixed acids in the wine, which is typically a combination of tartaric, malic, and citric acids\
>\
><b>Volatile Acidity</b>:       The amount of volatile acids in the wine, primarily acetic acid\
>\
><b>Citric Acid</b>:            The amount of citric acid in the wine, contributing to the overall acidity\
>\
><b>Residual Sugar</b>:         The amount of sugar remaining after fermentation\
>\
><b>Chlorides</b>:              The amount of chlorides in the wine, which can indicate the presence of salt\
>\
><b>Free Sulfar Dioxide</b>:    The amount of free sulfur dioxide in the wine, used as a preservative\
>\
><b>Total Sulfur Dioxide</b>:   The total amount of sulfur dioxide, including bound and free forms\
>\
><b>Density</b>:                The density of the wine, related to alcohol and sugar content\
>\
><b>PH</b>:                     The pH level of the wine, indicating its acidity\
>\
><b>Sulphates</b>:              The amount of sulphates in the wine, contributing to its taste and preservation\
>\
><b>Alcohol</b>: 	            The alcohol content of the wine in percentage

### Label:
><b>Quality</b>:              The quality level of the alcohol. Originally an integer from 3-9,\
>but processed into Low (3 & 4), Med (5-7), High (8 & 9)

### Results:
100% accuracy on Training Set, 72% accuracy on Validation Set.

Oddly, the 'Med' category was easiest to identify, while the 'Low' and 'High' categories had the highest amount of missclassifications.
Interestingly, the 'Low' and 'High' categories were more likely to missclassify themselves as being 'High' and 'Low' (respectively), instead of 'Med', which to me seems unintuitive.

<img src="https://github.com/user-attachments/assets/fd3d47d0-7f41-441d-84c3-16ac712e0cb6" width="600"></img>
<img src="https://github.com/user-attachments/assets/6996da47-5d6e-4f98-b949-651e21994a5e" width="600"></img>



Dataset Sourced from: https://www.kaggle.com/datasets/taweilo/wine-quality-dataset-balanced-classification
