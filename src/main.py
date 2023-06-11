""" AI4ALL -> A streamlit app to predict data for classification or regression tasks 
"""

import streamlit as st
import pandas as pd
from numpy import nan
from typing import List, Tuple, Dict, Union
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_squared_log_error, mean_absolute_percentage_error

# Constants
FILENAME = "prediction_results.csv"

# Functions
def disable_option_selection() -> None:
    if not st.session_state.run_model:
        st.session_state.run_model = True
    if st.session_state.run_model:
    # if True, sets to False, and vice-versa
        st.session_state.radio_disabled = True
        
def load_data(file: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(file)
    st.dataframe(df.head(), hide_index=True)
    return df

def get_features_names(input_feat_names: str, sep: str) -> List:
    return input_feat_names.split(sep=sep)

def get_X_y_data(data: pd.DataFrame, feature_cols: List, target_col: str=None):
    X = data[feature_cols]
    if target_col is not None:
        y = data[target_col]
        return X, y
    return X

def select_numerical_features(all_features, nominal_feats):
    feats_new = all_features[:]
    if nominal_feats is not None:
        for feat in nominal_feats:
            feats_new.remove(feat)
    return feats_new

def run_classifier(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    X_to_predict: pd.DataFrame, 
    nominal_cols: List[str]=None
) -> Tuple[List, str]:
    
    # Training step to get scores
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    features_list = select_numerical_features(
        all_features=X_train.columns.tolist(),
        nominal_feats=nominal_cols,
    )
    
    pipeline = make_pipeline(
        ColumnTransformer([
            ("ordinal_encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=nan), nominal_cols),
            ("scaler", StandardScaler(), features_list)
        ], remainder='passthrough') if nominal_cols is not None else None,
        StandardScaler(),
        RandomForestClassifier(random_state=42),
    )
 
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='macro')
    # report = classification_report(y_test, y_test_pred)
    report = {
        "accuracy": accuracy,
        "f1_score": f1,
    }
    
    # Predict the new data
    y_pred = pipeline.predict(X_to_predict)
    
    return y_pred, report

def run_regressor(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    X_to_predict: pd.DataFrame,
    nominal_cols: List[str]=None,
) -> Tuple[List, Dict]:

    # Training step to get scores
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    
    pipeline = make_pipeline(
        ColumnTransformer([
            (OrdinalEncoder(handle_unknown='use_encoded_value'), nominal_cols),
        ], remainder='passthrough') if nominal_cols is not None else None,
        StandardScaler(),
        HistGradientBoostingRegressor(random_state=42),
    )
    pipeline.fit(X_train, y_train)
    y_test_pred = pipeline.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test, y_test_pred)
    rmsle = mean_squared_log_error(y_test, y_test_pred, squared=False)
    
    report = {
        'MAPE': mape,
        'RMSLE': rmsle,
    }
    
    # Predict the new data
    y_pred = pipeline.predict(X_to_predict)
    
    return y_pred, report

def run_pipeline(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    X_to_predict: pd.DataFrame, 
    feature_cols: List[str],
    target_col: str,
    nominal_cols: List[str]=None,
    task: str='classification'
) -> str:
    
    # disable_option_selection()
    if task == 'classification':
        y_pred, report = run_classifier(X, y, X_to_predict, nominal_cols)
    else:
        y_pred, report = run_regressor(X, y, X_to_predict, nominal_cols)

    csv = save_prediction(X_to_predict, y_pred, feature_cols=feature_cols.tolist(), target_col=target_col)
    with st.empty():
        st.success("Data saved as CSV")
        st.write(report)
    return csv

@st.cache_data
def save_prediction(X, y_pred, feature_cols, target_col):
    df_pred = pd.DataFrame(X, columns=feature_cols)
    df_pred[target_col] = y_pred
    return df_pred.to_csv(index=False).encode('utf-8')


def show_error(steps):
    steps_incompleted = [str(step+1) for step, completed in enumerate(steps) if not completed]
    steps_incompleted = ", ".join(steps_incompleted)
    err_str = f"ERROR! Missing steps {steps_incompleted}"
    st.error(err_str, icon="🚨")

# MAIN FUNCTION
def main():

    # Store the initial value of widgets in session state
    if "radio_disabled" not in st.session_state:
        st.session_state.radio_disabled = False
    if "run_model" not in st.session_state:
        st.session_state.run_model = False

    st.write(
        """
            # :robot_face: PredictMe
            > A web Artificial Intelligence model for everyone - just upload your dataset and let it predict for you
        """
    )

    option = st.radio(
        "Select which problem you need to solve:", 
        ["Classification", "Regression"],
        disabled=st.session_state.radio_disabled,
    ) 

    # File Upload
    st.markdown('---') # line divisor
    st.markdown(
        """
            ### Upload your data
            :heavy_exclamation_mark: Requirements for the dataset:
            
            - Must be a CSV file
            - The data needs to have :blue[**at least one feature column**]
            - The data needs to have :blue[**only one target column**]
            - :red[Nominal categorical] features must be specified, if there are any - :blue[numerical categorical] don't need to be specified
            - Make sure data is not dirty (i.e.: wrong input values, incorrect data types, etc)
        """
    )

    steps = [False, False]

    train_file = st.file_uploader(":information_source: Step 1: :blue[**TRAIN DATA**] - Upload your CSV file here", type=["csv"])

    if train_file is not None:
        df_train = load_data(train_file)
        steps[0] = True

    test_file = st.file_uploader(
        ":information_source: Step 2: :blue[**TEST DATA**] - Upload your CSV file here. :violet[Obs: this is the data you **want** to predict]",
        type=["csv"],
    )

    if test_file is not None:
        df_test = load_data(test_file)
        steps[1] = True

    nominal_features_names = st.text_input(
        label=":information_source: Step 3 - OPTIONAL: If there are any nominal (string) features, provide the columns separeted by comma (e.g.: col1, col2, col3)",
        help="Press enter to send the text",
    )

    nominal_feat_names = None
    if nominal_features_names != "":
        nominal_feat_names = get_features_names(nominal_features_names, sep=',')

    if steps[0]:
        feat_names_list = df_train.columns[:-1]
        target_name = df_train.columns[-1]

    st.markdown(
        f"""
            ### All set?
            Then click on ```run {option.lower()} model``` to get the prediction for your dataset!
        """
    )

    # Separate data in train and test only if steps 1 and 2 are done
    if all(steps):
        X_train, y_train = get_X_y_data(df_train, feature_cols=feat_names_list, target_col=target_name)
        X_to_predict = get_X_y_data(df_test, feature_cols=feat_names_list)

    csv = ""
    clicked_on_run = st.button(f"Run {option.lower()} model")
    
    if clicked_on_run:
        if not all(steps):
            with st.empty():
                show_error(steps)
        else:
            csv = run_pipeline(
                X=X_train, 
                y=y_train,
                X_to_predict=X_to_predict,
                feature_cols=feat_names_list,
                target_col=target_name,
                nominal_cols=nominal_feat_names,
                task=option,
            )
            st.session_state.run_model = True

    st.download_button(
        "Save prediction",
        data=csv,
        mime='text/csv',
        file_name=FILENAME,
        disabled=not st.session_state.run_model,
    )
        
if __name__ == '__main__':
    main()