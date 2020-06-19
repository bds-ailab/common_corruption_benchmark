# Copyright 2020 BULL SAS All rights reserved #

# The range of possible amounts of the corruptions
corruption_amount = {}
corruption_amount["quantization"] = [8,3]
corruption_amount["gaussian"] = [0.05,0.21]
corruption_amount["salt_pepper"] = [0.002,0.04]
corruption_amount["blur"] = [0.4,1.0]
corruption_amount["low_res"] = [1.2,3.5]
corruption_amount["pixelate"] = [2,4]
corruption_amount["artifacts"] = [11,150]
corruption_amount["vertical_artifacts"]= [11,130]
corruption_amount["rhombus"] = [7,65]
corruption_amount["rain"] = [10,95]
corruption_amount["circle"] = [5,50]
corruption_amount["arlequin"] = [2,27]
corruption_amount["obstruction"] = [40,120]
corruption_amount["border"] = [6,42]
corruption_amount["translation"] = [8,66]
corruption_amount["shear"] = [10,34]
corruption_amount["elastic"] = [35,100]
corruption_amount["rotation"] = [6,45]
corruption_amount["backlight"] = [0.11,0.40]
corruption_amount["brightness"] = [0.16,0.50]
corruption_amount["contrast"] = [0.34,0.73]
corruption_amount["color_disortion"] = [0.08,0.40]
corruption_amount["gray_scale"] = [0.55,1] #only up to 74 rob
corruption_amount["hue"] = [0.06,0.5]  #only up to 63 rob

# Error rates for a standard resnet50. They are used to compute the mCE score.
ref_error = {}
ref_error["none"] = 0.239
ref_error["quantization"] = 0.445
ref_error["gaussian"] = 0.386
ref_error["salt_pepper"] = 0.459
ref_error["blur"] = 0.330
ref_error["low_res"] = 0.375
ref_error["pixelate"] = 0.612
ref_error["artifacts"] = 0.529
ref_error["vertical_artifacts"] = 0.487
ref_error["rhombus"] = 0.441
ref_error["rain"] = 0.474
ref_error["circle"] = 0.430
ref_error["arlequin"] = 0.340
ref_error["obstruction"] = 0.327
ref_error["border"] = 0.341
ref_error["translation"] = 0.300
ref_error["shear"] = 0.394
ref_error["elastic"] = 0.293
ref_error["rotation"] = 0.500
ref_error["backlight"] = 0.325
ref_error["brightness"] = 0.336
ref_error["contrast"] = 0.287
ref_error["color_disortion"] = 0.304
ref_error["gray_scale"] = 0.307
ref_error["hue"] = 0.385
