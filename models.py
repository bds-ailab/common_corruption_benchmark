""" Loader of the models used to estimate the correlations in terms of robustness between benchmarks """

import torchvision
import torch
import os
import antialiased_cnns

def load(model_name, model_path="Results/trained_models", nb_gpu=1):
    model_path = os.path.join(model_path,model_name)
    if os.path.exists(model_path):
        checkpoint_name = os.listdir(model_path)[0]
        model_path = os.path.join(model_path,checkpoint_name)


    # Plain models
    if model_name == "resnet50":
        classifier = torchvision.models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        classifier = torchvision.models.resnet18(pretrained=True)
    elif model_name == "densenet121":
        classifier = torchvision.models.densenet121(pretrained=True)
    elif model_name == "alexnet":
        classifier = torchvision.models.alexnet(pretrained=True)
    elif model_name == "resnet152":
        classifier = torchvision.models.resnet152(pretrained=True)
    elif model_name == "efficientnet_b0":
        classifier = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b0', pretrained=True)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
    elif model_name == "resnext101_32x8d":
        classifier = torchvision.models.resnext101_32x8d(pretrained=True)

    # Robust models
    elif model_name == "ANT":
        classifier = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['model_state_dict'])

    elif model_name == "SIN":
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['state_dict'])

    elif model_name == "Augmix":
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['state_dict'])

    elif model_name == "DeepAugment":
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['state_dict'])

    elif model_name == "Cutmix":
        classifier = torchvision.models.resnet152(pretrained=False)
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "FastAutoAugment":
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint)

    elif model_name == "RSC":
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(model_path)
        classifier.load_state_dict(checkpoint['state_dict'])

    elif model_name == "MoPro":
        classifier = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load(model_path)['state_dict']
        for k in list(checkpoint.keys()):
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                checkpoint[k[len("module.encoder_q."):]] = checkpoint[k]
            del checkpoint[k]
        checkpoint['fc.weight'] = checkpoint['classifier.weight']
        checkpoint['fc.bias'] = checkpoint['classifier.bias']
        del checkpoint['classifier.weight']
        del checkpoint['classifier.bias']
        classifier.load_state_dict(checkpoint)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))


    elif model_name == "AdvProp":
        classifier = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b0_ap', pretrained=True)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "AT_Linf_4":
        classifier = torchvision.models.resnet50(pretrained=False)
        checkpoint = torch.load(model_path)['model']
        for k in list(checkpoint.keys()):
            if 'module.model.' not in k:
                checkpoint.pop(k)
        checkpoint = {k.split('model.')[1]:v for k,v in checkpoint.items()}
        classifier.load_state_dict(checkpoint)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "NoisyStudent":
        classifier = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'tf_efficientnet_b0_ns', pretrained=True)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "SpatialAdv":
        classifier = torchvision.models.resnet18(pretrained=False)
        checkpoint = torch.load(model_path)['model']
        for k in list(checkpoint.keys()):
            if 'attacker' in k:
                checkpoint.pop(k)
        for k in list(checkpoint.keys()):
            if 'module.model.' not in k:
                checkpoint.pop(k)
        checkpoint = {k.split('model.')[1]:v for k,v in checkpoint.items()}
        classifier.load_state_dict(checkpoint)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "Anti_Alias":
        classifier = antialiased_cnns.densenet121(pretrained=True, filter_size=4)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "WSL":
        classifier = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    elif model_name == "SSL":
        classifier = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_ssl')
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))

    else:
        # Load models trained with data augmentation
        classifier = torchvision.models.resnet50(pretrained=False)
        classifier = torch.nn.DataParallel(classifier, device_ids=list(range(nb_gpu)))
        checkpoint = torch.load(os.path.join(model_path,'{}/checkpoint'.format(model_name)))
        classifier.load_state_dict(checkpoint)

    return classifier
