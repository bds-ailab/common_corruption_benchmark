# Copyright 2020 BULL SAS All rights reserved #

from torchvision import transforms
import torch
from random import randint, choice, random
import numpy as np
import os
from operator import add

# Pixel values of input images should be in the range [0-1] and RGB
class CC_Transform():
    def __init__(self,amount,corruption_kind):
        self.amount=amount
        null_filter = torch.zeros([1,1,3,3],dtype=torch.float32)
        one_filter = torch.ones([1,1,3,3],dtype=torch.float32)
        self.blur_filters = (1/9)*torch.cat([torch.cat([one_filter,null_filter,null_filter],1),torch.cat([null_filter,one_filter,null_filter],1),torch.cat([null_filter,null_filter,one_filter],1)],0)
        self.corruption_kind = corruption_kind

    def __call__(self,image):
        chosen_corruption = self.corruption_kind
        if chosen_corruption == "random":
            indice = randint(0,len(self.amount.keys())-1)
            chosen_corruption = list(self.amount.keys())[indice]

        if chosen_corruption == "none":
            return image

        elif chosen_corruption == "gaussian" :
            mean = torch.zeros([1])
            stddev = mean.clone().detach().uniform_(self.amount["gaussian"][0],self.amount["gaussian"][1])
            gaussian_sampler = torch.distributions.normal.Normal(mean,stddev)
            gaussian_mask =  torch.squeeze(gaussian_sampler.sample(image.shape))
            corrupted_image = image + torch.abs(gaussian_mask)
            corrupted_image = torch.clamp(corrupted_image,0,1)
            return corrupted_image

        elif chosen_corruption == "blur":
            blur_weight = torch.zeros([1]).uniform_(self.amount["blur"][0],self.amount["blur"][1])
            corrupted_image = image
            for i in range(5):
                corrupted_image = torch.squeeze(torch.nn.functional.conv2d(torch.unsqueeze(corrupted_image,0),self.blur_filters,padding=1))
            return torch.clamp((1-blur_weight)*image + blur_weight*corrupted_image,0,1)

        elif chosen_corruption == "contrast":
            image_contrast_loss = torch.zeros([1]).uniform_(self.amount["contrast"][0],self.amount["contrast"][1])
            corrupted_image = (1-image_contrast_loss)*image + image_contrast_loss*torch.mean(image)
            return corrupted_image

        elif chosen_corruption == "brightness":
            image_brightness_shift = (2*(torch.zeros([1]).bernoulli_())-1)*(torch.ones([1]).uniform_(self.amount["brightness"][0],self.amount["brightness"][1]))
            corrupted_image = image + image_brightness_shift
            corrupted_image = torch.clamp(corrupted_image,0,1)
            return corrupted_image

        elif chosen_corruption == "salt_pepper":
            prob_sp = torch.zeros([1]).uniform_(self.amount["salt_pepper"][0],self.amount["salt_pepper"][1])/2
            sp_mask = 2*(torch.distributions.categorical.Categorical(torch.tensor([prob_sp,1-2*prob_sp,prob_sp])).sample(image.shape)-1).float()
            corrupted_image = image + sp_mask
            corrupted_image = torch.clamp(corrupted_image,0,1)
            return corrupted_image

        elif chosen_corruption == "artifacts":
            if self.amount["artifacts"][0] == 0:
                return image
            artifacts = get_artifacts(nb_min_line=self.amount["artifacts"][0], nb_max_line=self.amount["artifacts"][1], nb_artifact_max=12, image_height=image.shape[1], image_width=image.shape[2])
            chosen_gray = random()
            corrupted_image = image
            corrupted_image[:,artifacts[:,1],artifacts[:,0]] = chosen_gray
            corrupted_image = torch.clamp(corrupted_image,0,1)
            corrupted_image = corrupted_image
            return corrupted_image

        elif chosen_corruption == "vertical_artifacts":
            if self.amount["vertical_artifacts"][0] == 0:
                return image
            artifacts = get_artifacts(nb_min_line=self.amount["vertical_artifacts"][0], nb_max_line=self.amount["vertical_artifacts"][1], nb_artifact_max=12, image_height=image.shape[1], image_width=image.shape[2])
            chosen_gray = random()
            corrupted_image = image
            corrupted_image[:,artifacts[:,0],artifacts[:,1]] = chosen_gray
            corrupted_image = torch.clamp(corrupted_image,0,1)
            corrupted_image = corrupted_image
            return corrupted_image

        elif chosen_corruption == "quantization":
            if self.amount["quantization"][0] == 0:
                return image
            nb_level = randint(self.amount["quantization"][1],self.amount["quantization"][0])
            corrupted_image = image*nb_level
            corrupted_image = torch.round(corrupted_image)
            corrupted_image = corrupted_image / nb_level
            return torch.clamp(corrupted_image,0,1)

        elif chosen_corruption == "color_disortion":
            channel_nb = torch.randint(0,3,[1],dtype=torch.long)
            null_channel = torch.zeros([1,image.shape[1],image.shape[2]])
            image_amount = torch.zeros([1,1,1]).uniform_(self.amount["color_disortion"][0],self.amount["color_disortion"][1]).repeat(1,image.shape[1],image.shape[2])
            if channel_nb == 0 :
                color_mask = torch.cat([image_amount,null_channel,null_channel],dim=0)
            if channel_nb == 1 :
                color_mask = torch.cat([null_channel,image_amount,null_channel],dim=0)
            if channel_nb == 2 :
                color_mask = torch.cat([null_channel,null_channel,image_amount],dim=0)
            corrupted_image = image + color_mask
            corrupted_image = torch.clamp(corrupted_image,0,1)
            return corrupted_image

        elif chosen_corruption == "obstruction":
            if self.amount["obstruction"][0] == 0:
                return image
            mask_size = randint(self.amount["obstruction"][0], self.amount["obstruction"][1])
            mask = [[randint(0,image.shape[1]-mask_size),randint(0,image.shape[2]-mask_size)]]
            for i in range(mask_size):
                for k in range(mask_size):
                    if (i!=0) or (k!=0) :
                        mask.append(list( map(add, mask[0], [i,k])))
            corrupted_image = image.clone().detach()
            mask = torch.tensor(mask)
            corrupted_image[:,mask[:,0],mask[:,1]] = random()
            return corrupted_image

        elif chosen_corruption == "border":
            if self.amount["border"][0] == 0:
                return image
            corrupted_image = image
            border_thickness = randint(self.amount["border"][0], self.amount["border"][1])
            for i in range(border_thickness):
                corrupted_image[:,i,:] = corrupted_image[:,:,i] = corrupted_image[:,:,image.shape[2]-(i+1)] = corrupted_image[:,image.shape[1]-(i+1),:] = 0.5
            return corrupted_image

        elif chosen_corruption == "rhombus":
            if self.amount["rhombus"][0] == 0:
                return image
            rhombus_size = 7
            nb_rhombus = randint(self.amount["rhombus"][0], self.amount["rhombus"][1])
            corrupted_image = image.clone().detach()
            # chosen_color = random()
            for r in range(nb_rhombus):
                places = [[randint(rhombus_size,image.shape[1]-rhombus_size),randint(rhombus_size,image.shape[2]-rhombus_size)]]
                for i in range(-rhombus_size,rhombus_size):
                    for k in range(-(rhombus_size-abs(i)),rhombus_size-abs(i)):
                        if (i!=0) or (k!=0) :
                            places.append(list( map(add, places[0], [i,k])))
                places = torch.tensor(places)
                # corrupted_image[:,places[:,0],places[:,1]] = chosen_color
                corrupted_image[:,places[:,0],places[:,1]] = random()
            return corrupted_image

        elif chosen_corruption == "arlequin":
            if self.amount["arlequin"][0] == 0:
                return image
            rhombus_size = 7
            nb_rhombus = randint(self.amount["arlequin"][0], self.amount["arlequin"][1])
            corrupted_image = image.clone().detach()
            chosen_color1, chosen_color2, chosen_color0 = random(), random(), random()
            for r in range(nb_rhombus):
                places = [[randint(rhombus_size,image.shape[1]-rhombus_size),randint(rhombus_size,image.shape[2]-rhombus_size)]]
                for i in range(-rhombus_size,rhombus_size):
                    for k in range(-(rhombus_size-abs(i)),rhombus_size-abs(i)):
                        if (i!=0) or (k!=0) :
                            places.append(list( map(add, places[0], [i,k])))
                places = torch.tensor(places)
                corrupted_image[0,places[:,0],places[:,1]] = chosen_color0
                corrupted_image[1,places[:,0],places[:,1]] = chosen_color1
                corrupted_image[2,places[:,0],places[:,1]] = chosen_color2
            return corrupted_image

        elif chosen_corruption == "circle":
            if self.amount["circle"][0] == 0:
                return image
            nb_circle = randint(self.amount["circle"][0], self.amount["circle"][1])
            corrupted_image = image.clone().detach()
            # chosen_color1, chosen_color2, chosen_color0 = random(), random(), random()
            for r in range(nb_circle):
                places = [[randint(7,image.shape[1]-7),randint(7,image.shape[2]-7)]]
                for i in range(-6,7):
                    if abs(i)==6:
                        k = 1
                    if abs(i)==5:
                        k = 3
                    if abs(i)==4:
                        k = 4
                    if abs(i)==3 or abs(i)==2:
                        k = 5
                    if abs(i)==1 or abs(i)==0:
                        k = 6
                    for k in range(-k,k+1):
                        if (i!=0) or (k!=0) :
                            places.append(list( map(add, places[0], [i,k])))
                places = torch.tensor(places)
                # corrupted_image[0,places[:,0],places[:,1]] = chosen_color0
                # corrupted_image[1,places[:,0],places[:,1]] = chosen_color1
                # corrupted_image[2,places[:,0],places[:,1]] = chosen_color2
                corrupted_image[:,places[:,0],places[:,1]] = random()
            return corrupted_image

        elif chosen_corruption == "rain":
            if self.amount["rain"][0] == 0:
                return image
            nb_circle = randint(self.amount["rain"][0], self.amount["rain"][1])
            corrupted_image = image.clone().detach()
            chosen_color1, chosen_color2, chosen_color0 = random(), random(), random()
            for r in range(nb_circle):
                places = [[randint(7,image.shape[1]-7),randint(7,image.shape[2]-7)]]
                for i in range(-6,7):
                    if abs(i)==6:
                        k = 1
                    if abs(i)==5:
                        k = 3
                    if abs(i)==4:
                        k = 4
                    if abs(i)==3 or abs(i)==2:
                        k = 5
                    if abs(i)==1 or abs(i)==0:
                        k = 6
                    for k in range(-k,k+1):
                        if (i!=0) or (k!=0) :
                            places.append(list( map(add, places[0], [i,k])))
                places = torch.tensor(places)
                corrupted_image[0,places[:,0],places[:,1]] = 0.5 + 0.5*corrupted_image[0,places[:,0],places[:,1]]
                corrupted_image[1,places[:,0],places[:,1]] = 0.5 + 0.5*corrupted_image[1,places[:,0],places[:,1]]
                corrupted_image[2,places[:,0],places[:,1]] = 0.5 + 0.5*corrupted_image[2,places[:,0],places[:,1]]
            return corrupted_image

        elif chosen_corruption == "rotation":
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            angle = (2*(torch.zeros([1]).bernoulli_())-1)*randint(self.amount["rotation"][0], self.amount["rotation"][1])
            # corrupted_image = transforms.functional.pad(corrupted_image, 11, padding_mode='reflect')
            corrupted_image = transforms.functional.affine(corrupted_image, angle, [0,0], 1, 0)
            # corrupted_image = transforms.functional.center_crop(corrupted_image, (image.shape[1],image.shape[2]))
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            return corrupted_image

        elif chosen_corruption == "translation":
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            h_pace = randint(self.amount["translation"][0], self.amount["translation"][1])*(2*(torch.zeros([1]).bernoulli_())-1)
            w_pace = randint(self.amount["translation"][0], self.amount["translation"][1])*(2*(torch.zeros([1]).bernoulli_())-1)
            # corrupted_image = transforms.functional.pad(corrupted_image, 11, padding_mode='reflect')
            corrupted_image = transforms.functional.affine(corrupted_image, 0, [h_pace,w_pace], 1, 0)
            # corrupted_image = transforms.functional.center_crop(corrupted_image, (image.shape[1],image.shape[2]))
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            return corrupted_image

        elif chosen_corruption == "shear":
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            shear = (2*(torch.zeros([1]).bernoulli_())-1)*randint(self.amount["shear"][0], self.amount["shear"][1])
            corrupted_image = transforms.functional.affine(corrupted_image, 0, [0,0], 1, shear=[shear,0])
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            return corrupted_image

        elif chosen_corruption == "pixelate":
            if self.amount["pixelate"][0] == 0 or self.amount["pixelate"][0]==1:
                return image
            amount = randint(self.amount["pixelate"][0],self.amount["pixelate"][1])
            stride = [amount,amount]
            kernel_size =  [amount,amount]
            corrupted_image = torch.nn.functional.avg_pool2d(image, kernel_size, stride=stride, padding=0)
            if amount%2 == 1:
                corrupted_image = torch.nn.functional.pad(torch.nn.functional.upsample_nearest(corrupted_image.permute(0,2,1), scale_factor=amount), (1,1), mode='reflect')
                corrupted_image = torch.nn.functional.pad(torch.nn.functional.upsample_nearest(corrupted_image.permute(0,2,1), scale_factor=amount), (1,1), mode='reflect')
            else:
                corrupted_image = torch.nn.functional.upsample_nearest(corrupted_image.permute(0,2,1), scale_factor=amount)
                corrupted_image = torch.nn.functional.upsample_nearest(corrupted_image.permute(0,2,1), scale_factor=amount)
            return corrupted_image

        elif chosen_corruption == "low_res" :
            if self.amount["low_res"][0] == 0:
                return image
            height,width = image.shape[1], image.shape[2]
            resize_factor = torch.zeros([1]).uniform_(self.amount["low_res"][0],self.amount["low_res"][1])
            image = transforms.functional.to_pil_image(image,mode="RGB")
            corrupted_image = transforms.functional.resize(image, [int(height/resize_factor),int(width/resize_factor)])
            corrupted_image = transforms.functional.resize(corrupted_image, [height, width])
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            return corrupted_image

        elif chosen_corruption == "gray_scale":
            gray_weight = torch.zeros([1]).uniform_(self.amount["gray_scale"][0],self.amount["gray_scale"][1])
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            corrupted_image = transforms.functional.to_grayscale(corrupted_image, num_output_channels=3)
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            corrupted_image = torch.clamp(corrupted_image*gray_weight + image*(1-gray_weight),0,1)
            return corrupted_image

        elif chosen_corruption == "hue":
            hue_factor= (2*(torch.zeros([1]).bernoulli_())-1)*torch.zeros([1]).uniform_(self.amount["hue"][0],self.amount["hue"][1])
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            corrupted_image = transforms.functional.adjust_hue(corrupted_image, hue_factor)
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            corrupted_image = torch.clamp(corrupted_image,0,1)
            return corrupted_image

        elif chosen_corruption == "elastic":
            height,width = image.shape[1], image.shape[2]
            axis = choice(["x","y"])
            shift = randint(self.amount["elastic"][0], self.amount["elastic"][1])
            corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
            if axis == "x":
                top,left = shift,0
                corrupted_image = transforms.functional.resized_crop(corrupted_image, shift, 0, height-shift, width, image.size()[1:])
            elif axis == "y":
                top,left = shift,0
                corrupted_image = transforms.functional.resized_crop(corrupted_image, 0, shift, height, width-shift, image.size()[1:])
            corrupted_image = transforms.functional.to_tensor(corrupted_image)
            return corrupted_image

        elif chosen_corruption == "backlight":
            height,width = image.shape[1], image.shape[2]
            axis = choice(["x","y"])
            if axis == "x":
                border = width//2 + randint(-width//5,width//5)
                corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
                right_part = transforms.functional.to_tensor(transforms.functional.crop(corrupted_image, 0, border, height, width-border))
                left_part = transforms.functional.to_tensor(transforms.functional.crop(corrupted_image, 0, 0, height, border))
                light_shift = (2*(torch.zeros([1]).bernoulli_())-1)*torch.ones([1]).uniform_(self.amount["backlight"][0],self.amount["backlight"][1])
                right_part = torch.clamp(right_part+light_shift,0,1)
                left_part = torch.clamp(left_part-light_shift,0,1)
                corrupted_image = torch.cat([left_part,right_part], dim=2)
            elif axis == "y":
                border = height//2 + randint(-height//5,height//5)
                corrupted_image = transforms.functional.to_pil_image(image,mode="RGB")
                down_part = transforms.functional.to_tensor(transforms.functional.crop(corrupted_image, border, 0, height-border, width))
                up_part = transforms.functional.to_tensor(transforms.functional.crop(corrupted_image, 0, 0, border, width))
                light_shift = (2*(torch.zeros([1]).bernoulli_())-1)*torch.ones([1]).uniform_(self.amount["backlight"][0],self.amount["backlight"][1])
                up_part = torch.clamp(up_part+light_shift,0,1)
                down_part = torch.clamp(down_part-light_shift,0,1)
                corrupted_image = torch.cat([up_part,down_part], dim=1)
            return corrupted_image



def select_lines(nb_min_line,nb_max_line,image_height):
    lines = []
    nb_lines = randint(nb_min_line,nb_max_line)
    for i in range(nb_lines):
        lines.append(randint(1,image_height-2))
    return lines

def initiate_artifact(num_line,image_width):
    # num_line = [num_line]*5
    initial_artifacts = list( map(add, [-2,-1,0,1,2], [randint(2,image_width-3)]*5) )
    initial_artifacts = [[elem,num_line] for elem in initial_artifacts]
    # initial_artifact = initial_artifact + [-2,-1,0,1,2]
    return initial_artifacts

def duplicate_artifacts(initial_artifacts,nb_artifact_max,image_width):
    initial_artifacts = np.asarray(initial_artifacts)
    up_shift = np.asarray([[8,0],[9,0],[10,0],[11,0],[12,0]])
    down_shift = np.asarray([[-8,0],[-7,0],[-6,0],[-5,0],[-4,0]])
    for k in range(initial_artifacts.shape[0]//5):
        nb_artifact = randint(0,nb_artifact_max//2)
        for i in range(-nb_artifact,nb_artifact):
            if i != 0 :
                if initial_artifacts[5*(k+1)-1][0] < image_width-12 :
                    initial_artifacts = np.concatenate([initial_artifacts,initial_artifacts[5*k:5*(k+1)] + up_shift],0)
                if initial_artifacts[5*k][0] > 7 :
                    initial_artifacts = np.concatenate([initial_artifacts,initial_artifacts[5*k:5*(k+1)] + down_shift],0)
    shift_line = np.full(initial_artifacts.shape,1)
    # print(up_line.shape)
    shift_line[:,0] = 0
    initial_artifacts = np.concatenate([initial_artifacts,initial_artifacts+shift_line,initial_artifacts-shift_line],axis=0)
    return initial_artifacts

def get_artifacts(nb_min_line,nb_max_line,nb_artifact_max,image_height,image_width):
    initial_artifacts = []
    lines_corrupted = select_lines(nb_min_line,nb_max_line,image_height)
    for line in lines_corrupted:
        initial_artifacts = initial_artifacts + initiate_artifact(line,image_width)
    # initial_artifacts = [initiate_artifact(elem,image_width) for elem in lines_corrupted]
    artifacts = duplicate_artifacts(initial_artifacts,nb_artifact_max,image_width)
    return artifacts
