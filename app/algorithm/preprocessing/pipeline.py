from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys, os
import joblib
import algorithm.preprocessing.preprocessors as preprocessors


PREPROCESSOR_FNAME = "preprocessor.save"


def get_preprocess_pipeline(pp_params, model_cfg):

    pp_step_names = model_cfg["pp_params"]["pp_step_names"]

    pipe_steps = []

    # ===== KEEP ONLY COLUMNS WE USE   =====
    pipe_steps.append(
        (
            pp_step_names["COLUMN_SELECTOR"],
            preprocessors.ColumnSelector(columns=pp_params["retained_vars"]),
        )
    )

    # ===============================================================
    # ===== NUMERICAL VARIABLES =====

    pipe_steps.append(
        # ===== CAST CAT VARS TO STRING =====
        (
            pp_step_names["FLOAT_TYPE_CASTER"],
            preprocessors.FloatTypeCaster(num_vars=pp_params["num_vars"]),
        )
    )

    # Standard Scale num vars
    pipe_steps.append(
        (
            pp_step_names["STANDARD_SCALER"],
            SklearnTransformerWrapper(
                StandardScaler(), variables=pp_params["num_vars"]
            ),
        )
    )

    # ===============================================================
    # X column selector
    pipe_steps.append(
        (
            pp_step_names["X_SPLITTER"],
            preprocessors.XSplitter(
                id_col=pp_params["id_field"],
            ),
        )
    )
    # ===============================================================

    pipeline = Pipeline(pipe_steps)

    return pipeline


def save_preprocessor(preprocess_pipe, file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    joblib.dump(preprocess_pipe, file_path_and_name)


def load_preprocessor(file_path):
    file_path_and_name = os.path.join(file_path, PREPROCESSOR_FNAME)
    preprocess_pipe = joblib.load(file_path_and_name)
    return preprocess_pipe
