from flask import Flask, request
import sys

import pip
from mushroom.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from mushroom.logger import logging
from mushroom.exception import MushroomException
import os, sys
import json
from mushroom.config.configuration import configuration
from mushroom.constant import CONFIG_DIR, get_current_time_stamp
from mushroom.pipeline.pipeline import Pipeline
from mushroom.entity.mushroom_predictor import MushroomPredictor, MushroomData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "mushroom"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from mushroom.logger import get_log_dataframe

MUSHROOM_DATA_KEY = "mushroom_data"
MEDIAN_MUSHROOM_VALUE_KEY = "class_"

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path': 'mushroom'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("mushroom", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=configuration(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logging.info("Prediction started")
    context = {
        MUSHROOM_DATA_KEY: None,
        MEDIAN_MUSHROOM_VALUE_KEY: None
    }

    if request.method == 'POST':
        cap_shape = str(request.form['cap_shape'])
        cap_surface = str(request.form['cap_surface'])
        cap_color = str(request.form['cap_color'])
        bruises = str(request.form['bruises'])
        odor = str(request.form['odor'])
        # gill_attachment = str(request.form['gill_attachment'])
        gill_spacing = str(request.form['gill_spacing'])
        gill_size = str(request.form['gill_size'])
        gill_color = str(request.form['gill_color'])
        # stalk_shape = str(request.form['stalk_shape'])  
        stalk_root = str(request.form['stalk_root'])
        stalk_surface_above_ring = str(request.form['stalk_surface_above_ring'])
        # stalk_surface_below_ring = str(request.form['stalk_surface_below_ring'])
        stalk_color_above_ring = str(request.form['stalk_color_above_ring'])
        # stalk_color_below_ring = str(request.form['stalk_color_below_ring'])
        # veil_type = str(request.form['veil_type'])
        # veil_color = str(request.form['veil_color'])
        # ring_number = str(request.form['ring_number'])
        ring_type = str(request.form['ring_type'])
        spore_print_color = str(request.form['spore_print_color'])
        population = str(request.form['population'])
        habitat = str(request.form['habitat'])



        mushroom_data = MushroomData(
                                    cap_shape = cap_shape,
                                    cap_surface = cap_surface,
                                    cap_color = cap_color,
                                    bruises = bruises,
                                    odor = odor,
                                    # gill_attachment = gill_attachment,
                                    gill_spacing = gill_spacing,
                                    gill_size = gill_size,
                                    gill_color = gill_color,
                                    # stalk_shape = stalk_shape,
                                    stalk_root = stalk_root,
                                    stalk_surface_above_ring = stalk_surface_above_ring,
                                    # stalk_surface_below_ring = stalk_surface_below_ring,
                                    stalk_color_above_ring = stalk_color_above_ring,
                                    # stalk_color_below_ring = stalk_color_below_ring,
                                    # veil_type = veil_type,
                                    # veil_color = veil_color,
                                    # ring_number = ring_number,
                                    ring_type = ring_type,
                                    spore_print_color = spore_print_color,
                                    population = population,
                                    habitat = habitat
                                   )
        mushroom_df = mushroom_data.get_mushroom_input_data_frame()
        logging.info("Prediction running")
        mushroom_predictor = MushroomPredictor(model_dir=MODEL_DIR)
        class_ = mushroom_predictor.predict(X=mushroom_df)
        context = {
            MUSHROOM_DATA_KEY: mushroom_data.get_mushroom_data_as_dict(),
            MEDIAN_MUSHROOM_VALUE_KEY: class_
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run()
