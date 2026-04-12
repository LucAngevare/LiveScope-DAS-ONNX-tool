# das_custom_objects.py
from das.kapre.time_frequency import Spectrogram, Melspectrogram
from das.kapre.utils import Normalization2D
from das.tcn import tcn as das_tcn

custom_objects = {
    "Spectrogram": Spectrogram,
    "Melspectrogram": Melspectrogram,
    "Normalization2D": Normalization2D,
    "das.tcn.tcn": das_tcn,
    "tcn": das_tcn,
}