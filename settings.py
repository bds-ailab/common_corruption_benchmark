# Copyright 2020 BULL SAS All rights reserved #

# The severity ranges of the used  common corruptions
corruption_amount = {}
corruption_amount["gaussian"] = [0.05,0.18]
corruption_amount["contrast"] = [0.33,0.74]
corruption_amount["brightness"] = [0.16,0.51]
corruption_amount["blur"] = [0.4,0.95]
corruption_amount["color_disortion"] = [0.09,0.40]
corruption_amount["obstruction"] = [47,125]
corruption_amount["rotation"] = [7,50]
corruption_amount["translation"] = [15,62]
corruption_amount["quantization"] = [8,3]
corruption_amount["shear"] = [12,39]
corruption_amount["pixelate"] = [2,4]
corruption_amount["gray_scale"] = [0.49,1] #only up to 74 rob
corruption_amount["elastic"] = [44,110]
corruption_amount["backlight"] = [0.11,0.44]
corruption_amount["salt_pepper"] = [0.003,0.032]
corruption_amount["uniform"] = [0.06,0.20]
corruption_amount["hue"] = [0.05,0.5]  #only up to 63 rob
corruption_amount["rain"] = [12,120]
corruption_amount["thumbnail_resize"] = [1.1,3.25]
corruption_amount["border"] = [9,46]
corruption_amount["artifacts"] = [14,170]
corruption_amount["vertical_artifacts"]= [14,180]
corruption_amount["rhombus"] = [9,76]
corruption_amount["circle"] = [7,50]
corruption_amount["arlequin"] = [3,36]

# Error rates of the torchvision pretrained ResNet50, on the ImageNet validation set corrupted with the proposed common corruptions
ref_error = {}
ref_error["none"] = 0.435
ref_error["quantization"] =  0.592
ref_error["gaussian"] = 0.620
ref_error["salt_pepper"] = 0.649
ref_error["blur"] = 0.570
ref_error["thumbnail_resize"] = 0.574
ref_error["pixelate"] =  0.760
ref_error["artifacts"] = 0.746
ref_error["vertical_artifacts"] =  0.758
ref_error["rhombus"] = 0.728
ref_error["rain"] = 0.748
ref_error["circle"] = 0.720
ref_error["arlequin"] = 0.747
ref_error["obstruction"] = 0.597
ref_error["border"] = 0.670
ref_error["translation"] = 0.603
ref_error["shear"] = 0.657
ref_error["elastic"] = 0.541
ref_error["rotation"] = 0.711
ref_error["backlight"] = 0.591
ref_error["brightness"] = 0.577
ref_error["contrast"] = 0.576
ref_error["color_disortion"] = 0.606
ref_error["gray_scale"] = 0.580
ref_error["hue"] = 0.684
