# Fabric-Defect-Detection-System
The Automated Fabric Defect Detection Tool helps textile manufacturers identify fabric defects (tears, spots, weave inconsistencies) using image processing techniques like edge detection, thresholding, and contour analysis. It automates inspection, improves accuracy, and reduces reliance on manual checks, enhancing quality control efficiency.

Team Names + IDs :
 - Ahmed hossam eldin nazmy 202201490
 - Ahmed Adel Ahmed Khalil 202202565
 - Ahmed Magdy Mahmoud El Khatib 202201913
 - Ibrahim medhat ibrahim mady 202204875
 - Khaled Walid Samir 202200533
 - Khaled Mohammed Mahmoud 202202643

Expected Results (if you run the code)
Since it uses synthetically generated data, the results can vary slightly depending on randomness. However, you can expect something like:

Training/Validation Accuracy (after 20 epochs):

Training Accuracy: ~98–100%
Validation Accuracy: ~90–98%

Training History:
Two plots saved and displayed:

training_history.png: Accuracy and loss curves

confusion_matrix.png: Visual representation of prediction correctness

 Classification Report:
plaintext
Copy
Edit
Classification Report:
              precision    recall  f1-score   support

        hole       1.00      1.00      1.00        10
   horizontal       1.00      1.00      1.00        10
    vertical       1.00      1.00      1.00        10
      normal       1.00      1.00      1.00        10

    accuracy                           1.00        40
   macro avg       1.00      1.00      1.00        40
weighted avg       1.00      1.00      1.00        40
 Model File Saved:
fabric_defect_model.h5: The trained model.
