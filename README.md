# Machine Learning for Cell Finder

- This is for cell finder in 4 classes

    1. EOSINOPHIL:  0
    2. LYMPHOCYTE:  1
    3. MONOCYTE:    2
    4. NEUTROPHIL:  3

- Running Material from Kaggle Competition
- Using Algorithm
    1. CNN
    2. MobileNetV2
    3. VGG16
# Data Consist
- Train (JPEG)
- Test  (JPEG)
- Label (.csv)

# CNN
    - Running Performance: Approximately 94%
# MobileNetV2
    - Running Performance: Approximately 92%
# VGG16
    - Running Performance: Approximately 94%

# Package Using
    #!conda install --yes tensorflow-datasets
    #!conda install --yes matplotlib
    #!conda install --yes seaborn
    #!conda install --yes scipy
    #!conda install -c conda-forge opencv
    #!conda install --yes glob
    #!conda install --yes scikit-image
    #!pip install opencv-python
    #!pip install imutils
    #!pip install kaggle --upgrade
    #!pip install tqdm

# Data Processing
    - OpenCV
    - imutils

# General information
    - Epoch = 30
    - Running weight from pre-trained model
    - Supervised learning

# plot information
    - placed on the code after running

# Additional experiement
    - Import Transformed Data on the analysis as rotate data before calculate
        - This result gets approximately 86%
    - Also, testing algorithm after fine-tuned VGG and fine-tuned mobileNetV2
        - Those show the result that approximately 92% above 