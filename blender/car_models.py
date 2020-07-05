#!/usr/bin/env python
"""
    Brief: Car model summary
    Author: wangpeng54@baidu.com
    Date: 2018/6/10
"""

from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple('Label', [

    'name'        , # The name of a car type
    'id'          , # id for specific car type
    'category'    , # The name of the car category, 'SUV', 'Sedan' etc
    'categoryId'  , # The ID of car category. Used to create ground truth images
                    # on category level.
    ])


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# categoryID: 1 --> sedan | categoryID: 2 --> SUV
models = [
    #                   name                    id       category  categoryId
    Label(           'mercedes-vito-van',         79,      'van',          3),

]


#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
car_name2id = {label.name: label for label in models}
car_id2name = {label.id: label for label in models}

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

