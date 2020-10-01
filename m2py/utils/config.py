"""
This file contains channel and figure information specific to each SPM technique.
The dictionary is structured as follows:

data_info = {
    "NAME":
        {
            "properties (names of readings)": list,
            "sample_area" (area of scan / sample): int,
            "pixel_size (in micro meters)": int,
        },
    }
"""

data_info = {
    "QNM": {
        "properties": ["Adhesion", "Deformation", "Dissipation", "LogDMT", "Height", "Stiffness"],
        "sample_area": 1,
        "pixel_size": 1,
    },
    "AMFM": {"properties": ["Height", "Deformation", "Youngs Modulus", "Phase"], "sample_area": 1, "pixel_size": 1},
    "cAFM": {"properties": ["Current", "Height"], "sample_area": 1, "pixel_size": 1},
    "Full_QNM": {
        "properties": ["Zscale", "Height", "PFE", "Stiffness", "LogDMT", "Adhesion", "Deformation", "Dissipation"],
        "sample_area": 1,
        "pixel_size": 1,
    },
    "OPV_QNM": {
        "properties": ["Zscale", "PFE", "Stiffness", "LogDMT", "Adhesion", "Deformation", "Dissipation", "Height"],
        "sample_area": 1,
        "pixel_size": 1,
    },
    "Raman": {"properties": ["R", "G", "B"], "sample_area": 1, "pixel_size": 1},
}
