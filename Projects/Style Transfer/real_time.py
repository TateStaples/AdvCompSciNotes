from VGG import Vgg16
from transformer import TransformerNet
import utils

import numpy as np

import os
import time

import torch
from torchvision import transforms, datasets


train = False


def train_style():
    # format data
    transform = transforms.Compose([transforms.Scale(train_size),
                                    transforms.CenterCrop(train_size),
                                    transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.mul(255))])
    train_dataset = datasets.ImageFolder(dataset_path, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    # create some important objects
    transformer = TransformerNet()
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(vgg_path, "vgg16.wieght")))

    optimizer = torch.optim.Adam(transformer.parameters(), learn_rate)
    loss_calculator = torch.nn.MSELoss()

    # load and format style image
    style_img = utils.load_img_as_tensor(f"styles/{style}.jpg")
    formatted_style = style_img.repeat(batch_size, 1, 1, 1)
    formatted_style = utils.preprocess_batch(formatted_style)
    style_variable = torch.autograd.Variable(formatted_style)
    style_variable = utils.subtract_imagenet_mean_batch(style_variable)

    # extract data from style
    features_style = vgg(style_variable)
    gram_style = [utils.gram(y) for y in features_style]

    # train
    for epoch in range(epochs):
        transformer.train()
        total_content_loss = total_style_loss = 0
        count = 0
        for batch_id, (x, _) in enumerate(train_loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = torch.autograd.Variable(utils.preprocess_batch(x))

            y = transformer(x)

            xc = torch.autograd.Variable(x.data.clone(), volatile=True)

            y = utils.subtract_imagenet_mean_batch(y)
            xc = utils.subtract_imagenet_mean_batch(xc)

            features_y = vgg(y)
            features_xc = vgg(xc)

            f_xc_c = torch.autograd.Variable(features_xc[1].data, requires_grad=False)

            content_loss = content_wight * loss_calculator(features_y[1], f_xc_c)

            style_loss = 0.
            for m in range(len(features_y)):
                gram_s = torch.autograd.Variable(gram_style[m].data, requires_grad=False)
                gram_y = utils.gram(features_y[m])
                style_loss += style_weight * loss_calculator(gram_y, gram_s[:n_batch, :, :])

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            total_content_loss += content_loss.data[0]
            total_style_loss += style_loss.data[0]
    transformer.eval()
    transformer.cpu()
    save_name = f"epoch+{epochs}_{content_wight}_{style_weight}.model"
    save_path = os.path.join(f"saved-models/{style}.pth", save_name)
    torch.save(transformer.state_dict(), save_path)
    print("\nDone, trained model saved at", save_path)


def apply_style():
    t = time.time()
    content_image = utils.tensor_load_rgbimage(content_path)
    content_image = content_image.unsqueeze(0)
    print(f"format image: {t-time.time()}")
    with torch.no_grad():
        content_image = torch.autograd.Variable(utils.preprocess_batch(content_image))
        print(f"to variable: {time.time()-t}")
        output = style_model(content_image)
        utils.tensor_save_rgbimage(output, output_path)
        # img = utils.tensor_to_img(output)
        # return img
    print(f"final = {time.time()-t}")


def style_img(img, model):
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).float().unsqueeze(0)
    with torch.no_grad():
        content_image = torch.autograd.Variable(utils.preprocess_batch(tensor))
        output = model(content_image)
    img = output.clone().clamp(0, 255).numpy()
    img = np.transpose(img, (0, 2, 3, 1))[0]
    return img


# training configs - only used on train
batch_size = 4
epochs = 2
dataset_path = None
train_size = 256
log_interval = 500

content_wight = 1
style_weight = 5
learn_rate = .001


# models
vgg_path = ""

# images
style = "udnie"
content_path = "source.png"  # only used on apply
output_path = "output/test.png"  # only used on apply


if __name__ == '__main__':
    if not train:
        style_model = TransformerNet()
        style_model.load_state_dict(torch.load(f"saved-models/{style}.pth"))

    if train:
        train_style()
    else:
        apply_style()


