# my_custom_component/sensor.py

from homeassistant.components.sensor import SensorEntity
import voluptuous as vol
from homeassistant.helpers.event import (
    async_track_state_change,
    async_track_time_interval,
)
from .trainer import train
import logging
import os
import torch

current_dir = os.getcwd()

_LOGGER = logging.getLogger(__name__)

DOMAIN = "nn_service"

CONFIG_SCHEMA = vol.Schema(
    {
        vol.Required("platform"): DOMAIN,
        vol.Required("name"): str,
        vol.Required("input_entity_ids"): [str],
        vol.Required("label_entity_ids"): [str],
    },
    extra=vol.ALLOW_EXTRA,
)


def setup_platform(hass, config, add_entities, discovery_info=None):
    name = config["name"]
    input_entity_ids = config["input_entity_ids"]
    label_entity_ids = config["label_entity_ids"]
    add_entities([TensorSensor(hass, name, input_entity_ids, label_entity_ids)])


class TensorSensor(SensorEntity):
    def __init__(self, hass, name, input_entity_ids, label_entity_ids):
        self.hass = hass
        self._name = name
        self._input_entity_ids = input_entity_ids
        self._label_entity_ids = label_entity_ids
        self._attr_extra_state_attributes = {
            "inputs": input_entity_ids,
            "labels": label_entity_ids,
        }
        self._state = None

    def _state_listener(self, entity, old_state, new_state):
        """Called when the target device changes state."""
        _LOGGER.info("Entity: %s, old: %s, new: %s", entity, old_state, new_state)
        self.hass.async_add_job(self.async_update)

    async def async_added_to_hass(self):
        """Run when entity about to be added."""
        await super().async_added_to_hass()

        # Add listener
        async_track_state_change(
            self.hass, self._input_entity_ids, self._state_listener
        )

    # async def async_get_state() -> None:
    #     """Get the state of the device."""
    #     try:
    #         await self.state
    #     except:
    #         raise "Error getting tensor state"

    async def async_update(self):
        model_path = f"{current_dir}/config/custom_components/nn_service/{self.name}.pt"
        if os.path.exists(model_path):
            # Load the model
            model = torch.load(model_path)
            input_states = [
                self.hass.states.get(entity_id).state
                for entity_id in self._input_entity_ids
            ]
            input_states = [float(state) for state in input_states]
            predicted = model(torch.tensor(input_states))
            self._state = predicted.item()
            _LOGGER.info("Model Prediction: %s", predicted.item())

        self.async_schedule_update_ha_state()

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state
