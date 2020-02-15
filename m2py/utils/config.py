"""
This file contains channel and figure information specific to each SPM technique. The dictionary is structured as follows:

data_info{
    "SPM_technique":{
        "properties (i.e. channels)": ["List", "of", "Channels"],
        "sample_size (i.e. scan dimensions)": {
            "sample_key_word": int,
            "sample_key_word": int,
            "sample_key_word": int,
        },
    },
    ...
}
"""

data_info = {
    "QNM": {
        "properties": ["Adhesion", "Deformation", "Dissipation", "LogDMT", "Height", "Stiffness"],
        "sample_size": {"Backgrounded": 2, "2ComponentFilms": 0.5, "Nanowires": 5},
    },
    "AMFM": {"properties": ["Height", "Deformation", "Youngs Modulus", "Phase"], "sample_size": {"Nanowires": 2}},
    "cAFM": {"properties": ["Current", "Height"], "sample_size": {"Nanowires": 2}},
    "NEW": {
        "properties": ["Zscale", "PFE", "Stiffness", "LogDMT", "Adhesion", "Deformation", "Dissipation", "Height"],
        "sample_size": {"NA": 1},
    },
}
