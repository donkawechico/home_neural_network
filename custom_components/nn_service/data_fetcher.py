# import pandas as pd
import pdb
from datetime import datetime, timedelta
import logging
import homeassistant.core as core
from homeassistant.components.recorder.history import get_significant_states
from homeassistant.util import dt as dt_util

_LOGGER = logging.getLogger(__name__)


def fetch_data(hass: core.HomeAssistant, entity_ids):
    end_time = dt_util.utcnow()
    start_time = end_time - timedelta(days=7)

    states = get_significant_states(
        hass=hass,
        start_time=start_time,
        end_time=end_time,
        entity_ids=entity_ids,
    )
    return states
