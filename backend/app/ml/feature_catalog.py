from __future__ import annotations

FEATURE_DEFINITIONS: dict[str, dict[str, object]] = {
    "cap_shape": {
        "description": "Overall shape of the mushroom cap",
        "options": {
            "b": "Bell",
            "c": "Conical",
            "x": "Convex",
            "f": "Flat",
            "k": "Knobbed",
            "s": "Sunken",
        },
    },
    "cap_surface": {
        "description": "Texture of the mushroom cap surface",
        "options": {
            "f": "Fibrous",
            "g": "Grooves",
            "y": "Scaly",
            "s": "Smooth",
        },
    },
    "cap_color": {
        "description": "Primary color of the mushroom cap",
        "options": {
            "n": "Brown",
            "b": "Buff",
            "c": "Cinnamon",
            "g": "Gray",
            "r": "Green",
            "p": "Pink",
            "u": "Purple",
            "e": "Red",
            "w": "White",
            "y": "Yellow",
        },
    },
    "bruises": {
        "description": "Whether bruises are visible on the mushroom",
        "options": {
            "t": "Bruises visible",
            "f": "No bruises",
        },
    },
    "odor": {
        "description": "Main odor detected from the mushroom",
        "options": {
            "a": "Almond",
            "l": "Anise",
            "c": "Creosote",
            "y": "Fishy",
            "f": "Foul",
            "m": "Musty",
            "n": "None",
            "p": "Pungent",
            "s": "Spicy",
        },
    },
    "gill_attachment": {
        "description": "How the gills attach to the stalk",
        "options": {
            "a": "Attached",
            "d": "Descending",
            "f": "Free",
            "n": "Notched",
        },
    },
    "gill_spacing": {
        "description": "Spacing between gills",
        "options": {
            "c": "Close",
            "w": "Crowded",
            "d": "Distant",
        },
    },
    "gill_size": {
        "description": "Relative gill size",
        "options": {
            "b": "Broad",
            "n": "Narrow",
        },
    },
    "gill_color": {
        "description": "Color of the gills",
        "options": {
            "k": "Black",
            "n": "Brown",
            "b": "Buff",
            "h": "Chocolate",
            "g": "Gray",
            "r": "Green",
            "o": "Orange",
            "p": "Pink",
            "u": "Purple",
            "e": "Red",
            "w": "White",
            "y": "Yellow",
        },
    },
    "stalk_shape": {
        "description": "Shape of the stalk",
        "options": {
            "e": "Enlarging",
            "t": "Tapering",
        },
    },
    "stalk_root": {
        "description": "Root structure of the stalk",
        "options": {
            "b": "Bulbous",
            "c": "Club",
            "u": "Cup",
            "e": "Equal",
            "z": "Rhizomorphs",
            "r": "Rooted",
            "?": "Missing/unknown",
        },
    },
    "stalk_surface_above_ring": {
        "description": "Stalk surface texture above the ring",
        "options": {
            "f": "Fibrous",
            "y": "Scaly",
            "k": "Silky",
            "s": "Smooth",
        },
    },
    "stalk_surface_below_ring": {
        "description": "Stalk surface texture below the ring",
        "options": {
            "f": "Fibrous",
            "y": "Scaly",
            "k": "Silky",
            "s": "Smooth",
        },
    },
    "stalk_color_above_ring": {
        "description": "Stalk color above the ring",
        "options": {
            "n": "Brown",
            "b": "Buff",
            "c": "Cinnamon",
            "g": "Gray",
            "o": "Orange",
            "p": "Pink",
            "e": "Red",
            "w": "White",
            "y": "Yellow",
        },
    },
    "stalk_color_below_ring": {
        "description": "Stalk color below the ring",
        "options": {
            "n": "Brown",
            "b": "Buff",
            "c": "Cinnamon",
            "g": "Gray",
            "o": "Orange",
            "p": "Pink",
            "e": "Red",
            "w": "White",
            "y": "Yellow",
        },
    },
    "veil_type": {
        "description": "Type of veil surrounding the mushroom",
        "options": {
            "p": "Partial",
            "u": "Universal",
        },
    },
    "veil_color": {
        "description": "Color of the veil",
        "options": {
            "n": "Brown",
            "o": "Orange",
            "w": "White",
            "y": "Yellow",
        },
    },
    "ring_number": {
        "description": "Number of rings on the stalk",
        "options": {
            "n": "None",
            "o": "One",
            "t": "Two",
        },
    },
    "ring_type": {
        "description": "Shape/type of the ring",
        "options": {
            "c": "Cobwebby",
            "e": "Evanescent",
            "f": "Flaring",
            "l": "Large",
            "n": "None",
            "p": "Pendant",
            "s": "Sheathing",
            "z": "Zone",
        },
    },
    "spore_print_color": {
        "description": "Color of spore print",
        "options": {
            "k": "Black",
            "n": "Brown",
            "b": "Buff",
            "h": "Chocolate",
            "r": "Green",
            "o": "Orange",
            "u": "Purple",
            "w": "White",
            "y": "Yellow",
        },
    },
    "population": {
        "description": "Population density where mushroom is found",
        "options": {
            "a": "Abundant",
            "c": "Clustered",
            "n": "Numerous",
            "s": "Scattered",
            "v": "Several",
            "y": "Solitary",
        },
    },
    "habitat": {
        "description": "Natural habitat where mushroom is found",
        "options": {
            "g": "Grasses",
            "l": "Leaves",
            "m": "Meadows",
            "p": "Paths",
            "u": "Urban",
            "w": "Waste",
            "d": "Woods",
        },
    },
}

EXPECTED_FEATURES = list(FEATURE_DEFINITIONS.keys())
NUMERIC_FEATURES: list[str] = []
CATEGORICAL_FEATURES = EXPECTED_FEATURES

TARGET_LABELS = {
    0: "Likely Edible",
    1: "Likely Poisonous",
}

