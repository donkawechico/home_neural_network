import pdb
import os
import logging
import torch
import numpy as np
import math
from sklearn.model_selection import train_test_split
from .data_fetcher import fetch_data
import pandas as pd
from .trainer import train

DOMAIN = "nn_service"

_LOGGER = logging.getLogger(__name__)
current_dir = os.getcwd()


def predict(model, X_test):
    with torch.no_grad():
        outputs = model(X_test)
        return outputs


def setup(hass, config):
    """Set up is called when Home Assistant is loading our component."""

    def handle_hello(call):
        """Handle the service call."""
        entity = call.data["entity_id"][0]
        tensor_sensor_entity = hass.states.get(entity)

        # input_entity_ids = tensor_sensor_entity.attributes["inputs"]

        # Check if model file exists
        model_path = f"{current_dir}/config/custom_components/nn_service/{tensor_sensor_entity.name}.pt"
        # if os.path.exists(model_path):
        #     # Load the model
        #     model = torch.load(model_path)
        #     input_states = [
        #         hass.states.get(entity_id).state for entity_id in input_entity_ids
        #     ]
        #     input_states = [float(state) for state in input_states]
        #     predicted = model(torch.tensor(input_states))
        #     _LOGGER.info("Model Prediction: %s", predicted.item())
        # else:
        # Save the trained model
        model = train(hass, tensor_sensor_entity)
        torch.save(model, model_path)

        hass.states.set("nn_service.status", "Training completed successfully")

    hass.services.register(DOMAIN, "train", handle_hello)

    # Return boolean to indicate that initialization was successful.
    return True
