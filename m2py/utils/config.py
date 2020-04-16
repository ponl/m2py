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
    "Full_QNM": {
        "properties": ["Zscale", "Height", "PFE", "Stiffness", "LogDMT", "Adhesion", "Deformation", "Dissipation"],
        "sample_size": {"P3HT:PCBM_OPV": 0.5, "P3HT_OFET": 0.5},
    },
    "OPV_QNM": {
        "properties": ["Zscale", "PFE", "Stiffness", "LogDMT", "Adhesion", "Deformation", "Dissipation", "Height"],
        "sample_size": {"P3HT:PCBM_OPV": 0.5, "P3HT_OFET": 0.5},
    },
    "Raman": {"properties": ["R", "G", "B"], "sample_size": {"Microplastics": 20}},
}
