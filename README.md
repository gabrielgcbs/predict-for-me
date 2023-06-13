# 🤖 Predict4me
A web Artificial Intelligence model for everyone - just upload your dataset and let it predict for you

---

ℹ️ **ATTENTION**: The fork on [@gcbsdev](https://github.com/gcbsdev) is meant to be read-only and its purpose is for streaming the app to [Streamlit Community Cloud](https://streamlit.io/cloud). When submitting a Pull Request or opening and Issue, please always refer to the repo in [@gabrielgcbs](https://github.com/gabrielgcbs/predict-for-me)

## ❓ What is Predict4me?
Predict4me is a web Machine Learning app built with Python and [Streamlit](https://streamlit.io)  
  
It implements a simple user-friendly interface that let's you upload your data and click one button to train a model and predict the data.

> Note that the purporse of Predict4me is to help those who are not experienced with Machine Learning predict their data without needing to code a model, thus it is not suited for more complex predictions or model engineering.

Currently, the app covers only the two Supervised Learning problems, which are:
- Classification
- Regression

The models implemented are:
- Random Forest (for classification tasks)
- Histogram-based Gradient Boosting Regression Tree (for regression tasks)

ℹ️ For more information about how these models work, please refer to scikit-learn official documentation: [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier), [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor)

## 🏁 Quick Start

### 🏎️ How to run the app on your browser?

- Visit [predict4me app](https://gcbsdev-predict-for-me.streamlit.app/)
- Select the type of task you wish: **Classification** or **Regression** (not sure which one to choose? Have a look at [this tutorial](https://machinelearningmastery.com/classification-versus-regression-in-machine-learning/))
- Upload your data:
  - The requirements to get the model running on your data are:
    - File format: csv
    - The data needs to have at least one feature column
    - The data needs to have only one target column (this may change in the future)
    - Nominal categorical features **must** be specified, if there are any
    - Numerical categorical features **do not** need to be specified
    - It is advised to clean the data before jumping on the model
  - You must submit one file for training the model (step 1), and one file containg the actual data you want to predict (step 2)
    - For example, if you want to predict the house pricing on a city, provide a dataset with information about houses in the area and their prices on step 1, and a dataset with the information about the house you want to predict on step 2
- If there are any nominal features (i.e. *strings*), provide their names on step 3
- Click on *RUN MODEL*
- Save the results as a csv file

### 💻 How to install and run the app locally?
Install Python >= 3.9  

Clone the repository:
``` shell
$ git clone https://github.com/gabrielgcbs/predict-for-me
```

ℹ️ Recommended: Create and activate a virtual environment

On Windows:  
``` shell
C:\> python -m venv .venv
C:\> .venv\Scripts\activate.bat
```

On Linux:
``` shell
$ python -m venv .venv
$ source .venv/bin/activate
```

Install the dependencies:
``` shell
$ pip install requirements.txt
```

Run the app:
``` shell
$ streamlit run src/main.py
```

### ❤️ Guidelines for contributing

Contributions are welcomed. For that, please follow this simple guidelines:
- Check if there is already an issue opened
- If not, create an issue to describe your request and specify a label for it, e.g *bug*, *feature* or *documentation*
- If you want to make some changes on the code, create a pull request and link it to an existing issue:
  - Documentation is important, so for each significant changes made on the code, please provide a documentation about it, both in-code and on the pull request itself
