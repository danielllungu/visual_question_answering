import matplotlib.pyplot as plt
import torch
import numpy as np
import timm
import torchvision.transforms as transforms
import torch.nn as nn


class ImageFeatures:
    def __init__(self, model_name='resnet50', height=224, width=224, num_channels=3, norm_mean=None, norm_std=None, do_norm=False, is_transformer=False):
        if norm_std is None:
            norm_std = [0.229, 0.224, 0.225]
        if norm_mean is None:
            norm_mean = [0.485, 0.456, 0.406]

        self.num_channels = num_channels
        self.is_transformer = is_transformer
        self.do_norm = do_norm
        self.model = timm.create_model(model_name, pretrained=True, features_only=True).to(torch.device('cuda:0'))
        self.transformer_model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0,
                                                   global_pool='').to(torch.device('cuda:0'))
        self.transformer_model.eval()
        # self.model = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        self.resize = transforms.Resize((height, width))
        self.normalize = transforms.Normalize(mean=norm_mean,
                                              std=norm_std)
        self.to_tensor = transforms.ToTensor()
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1)).to(torch.device('cuda:0'))

    def get_features_img2vec(self, image):
        """
        :param image: PIL Image
        :return: Tensor: features of last layer of the model
        """
        image = self.resize(image)
        if self.do_norm:
            image = self.to_tensor(image)
            image = self.normalize(image).unsqueeze(0)
        else:
            image = self.to_tensor(image).unsqueeze(0)

        if self.is_transformer:
            out = self.transformer_model(image.to(torch.device('cuda:0')))

        else:
            self.model.eval()
            outputs = self.model(image.to(torch.device('cuda:0')))
            out = self.adaptive_avg_pool(outputs[-1])

        return out.detach().cpu()

    def get_features_img2vec_map(self, image):
        """
        :param image: PIL Image
        :return: Tensor: features of last layer of the model
        """
        image = self.resize(image)
        if self.do_norm:
            image = self.to_tensor(image)
            image = self.normalize(image).unsqueeze(0)
        else:
            image = self.to_tensor(image).unsqueeze(0)

        if self.is_transformer:
            out = self.transformer_model(image.to(torch.device('cuda:0')))

        else:
            self.model.eval()
            outputs = self.model(image.to(torch.device('cuda:0')))
            out = outputs[-1]

        return out.detach().cpu()

    def show_features(self, image):
        """

        :param image: PIL Image
        :return: Plot Images Features
        """
        image = self.resize(image)
        image = torch.as_tensor(np.array(image, dtype=np.float32)).transpose(2, 0)[None]
        image = self.normalize(image)

        print("Image shape", image.shape)
        outputs = self.model(image)

        print("Features shapes")
        for o in outputs:
            print(o.shape)
            plt.imshow(o[0].transpose(0, 2).sum(-1).detach().numpy())
            plt.show()

    def list_models(self, model_name):
        """

        :param model_name: string model name
        :return: list of Timm  models
        """
        print(timm.list_models('*' + model_name + '*', pretrained=True))
