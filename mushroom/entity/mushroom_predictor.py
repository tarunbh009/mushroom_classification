import os
import sys

from mushroom.exception import MushroomException
from mushroom.util.util import load_object

import pandas as pd
import numpy as np


class MushroomData:

    def __init__(self,
                cap_shape :str,
                cap_surface : str,
                cap_color : str,
                bruises : str,
                odor : str,
                # gill_attachment : str,
                gill_spacing : str,
                gill_size : str,
                gill_color : str,
                # stalk_shape : str,
                stalk_root : str,
                stalk_surface_above_ring : str,
                # stalk_surface_below_ring : str,
                stalk_color_above_ring : str,
                # stalk_color_below_ring : str,
                # veil_type : str,
                # veil_color : str,
                # ring_number : str,
                ring_type  : str,
                spore_print_color : str,
                population : str,
                habitat : str,
                class_ : str = None
                 ):
        try:
            self.cap_shape = cap_shape
            self.cap_surface = cap_surface
            self.cap_color = cap_color
            self.bruises = bruises
            self.odor = odor
            # self.gill_attachment = gill_attachment
            self.gill_spacing = gill_spacing
            self.gill_size = gill_size
            self.gill_color = gill_color
            # self.stalk_shape = stalk_shape
            self.stalk_root = stalk_root
            self.stalk_surface_above_ring =stalk_surface_above_ring
            # self.stalk_surface_below_ring =stalk_surface_below_ring
            self.stalk_color_above_ring = stalk_color_above_ring
            # self.stalk_color_below_ring = stalk_color_below_ring
            # self.veil_type = veil_type
            # self.veil_color = veil_color
            # self.ring_number = ring_number
            self.ring_type = ring_type
            self.spore_print_color = spore_print_color
            self.population = population
            self.habitat = habitat
            self.class_ = class_
        except Exception as e:
            raise MushroomException(e, sys) from e

    def get_mushroom_input_data_frame(self):

        try:
            mushroom_input_dict = self.get_mushroom_data_as_dict()
            return pd.DataFrame(mushroom_input_dict)
        except Exception as e:
            raise MushroomException(e, sys) from e

    def get_mushroom_data_as_dict(self):
        try:
            input_data = {
                "cap-shape" :  [self.cap_shape],
                "cap-surface" : [self.cap_surface],
                "cap-color" : [self.cap_color],
                "bruises" : [self.bruises],
                "odor"   :  [self.odor],
                # "gill-attachment" :  [self.gill_attachment],
                "gill-spacing" :  [self.gill_spacing],
                "gill-size" :[self.gill_size],
                "gill-color"    : [self.gill_color],
                # "stalk-shape"  : [self.stalk_shape],
                "stalk-root" : [self.stalk_root], 
                "stalk-surface-above-ring" : [self.stalk_surface_above_ring],
                # "stalk-surface-below-ring" : [self.stalk_surface_below_ring],
                "stalk-color-above-ring" :   [self.stalk_color_above_ring],
                # "stalk-color-below-ring"  :  [self.stalk_color_below_ring],
                # "veil-type" : [self.veil_type],
                # "veil-color" :   [self.veil_color],
                # "ring-number": [self.ring_number],
                "ring-type" : [self.ring_type],
                "spore-print-color" : [self.spore_print_color],
                "population" : [self.population],
                "habitat"     : [self.habitat]
                            }
            return input_data
        except Exception as e:
            raise MushroomException(e, sys)


class MushroomPredictor:

    def __init__(self, model_dir: str):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise MushroomException(e, sys) from e

    def get_latest_model_path(self):
        try:
            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)
            return latest_model_path
        except Exception as e:
            raise MushroomException(e, sys) from e

    def predict(self, X):
        try:
            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            predicted_value = model.predict(X)
            # reverse_mapping = {1: 'p', 0: 'e'}
            # median_mushroom_value=reverse_mapping[(round(median_mushroom_value[0]))]
            threshold = 0.5
            predicted_value_arr = np.where(predicted_value >= threshold, 'p', 'e')
            median_mushroom_value = predicted_value_arr[0]
            return median_mushroom_value
        except Exception as e:
            raise MushroomException(e, sys) from e