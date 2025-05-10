import os
import numpy as np
import sys
sys.path.insert(0,'pulp-frontnet/PyTorch/')
from Frontnet.Frontnet import FrontnetModel

sys.path.insert(0,'.')
from src.util import load_model, load_quantized

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def annotate_images_single(image_path, model_path, camera_intrinsics, camera_extrinsics, save_path='misc/annotated_img/', config='160x32', quantized=False):

    if not quantized:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if not quantized:
        print("loading full precision")
        model = load_model(path=model_path, device=device, config=config)
    else:
        print("loading quantized")
        model = load_quantized(path=model_path, config=config)
    
    model.eval()

    
    images = []
    for filename in sorted(os.listdir(image_path)):
        if filename.endswith('.npy'):
            images.append(np.load(image_path+filename))

    images = torch.tensor(np.array(images)).unsqueeze(1).float()

    predictions = torch.stack(model(images)).squeeze(2).mT.detach().numpy()


    if not quantized:
        save_path += 'full_precision/'
    else:
        save_path += 'quantized/'
    os.makedirs(save_path, exist_ok = True)

    np.save(save_path+'predictions.npy', predictions)

    ones = np.ones((1))


    for count, prediction in enumerate(tqdm(predictions)):
        u, v, w = camera_intrinsics @ camera_extrinsics @ np.concatenate((prediction[:3], ones))
        img_x = u/w
        img_y = v/w


        plt.tight_layout()
        plt.imshow(images[count][0], cmap='gray')
        plt.axis('off')
        plt.scatter(img_x, img_y, s=(0.01 + (500-0.01) * (2.5-prediction[0])))
        plt.annotate("{:.2f},  {:.2f}, {:.2f}".format(prediction[0], prediction[1], prediction[3]), (img_x, img_y))
        plt.savefig(save_path+f"img_{count:05d}.jpg", dpi=200)
        plt.close()


def annotate_images_both(image_path, model_path, quantized_path, camera_intrinsics, camera_extrinsics, save_path='misc/annotated_img/both/', config='160x32'):
    device = torch.device('cpu')
    model = load_model(path=model_path, device=device, config=config)
    model_q = load_quantized(path=quantized_path, config=config)

    model.eval()
    model_q.eval()

    images = []
    for filename in sorted(os.listdir(image_path)):
        if filename.endswith('.npy'):
            images.append(np.load(image_path+filename))

    images = torch.tensor(np.array(images)).unsqueeze(1).float()

    predictions = torch.stack(model(images.clone())).squeeze(2).mT.detach()#.numpy()
    predictions_q = torch.stack(model_q(images.clone())).squeeze(2).mT.detach()#.numpy()

    print(torch.nn.functional.mse_loss(predictions, predictions_q))

    os.makedirs(save_path, exist_ok = True)

    predictions = predictions.numpy()
    predictions_q = predictions_q.numpy()

    np.save(save_path+'predictions.npy', predictions)
    np.save(save_path+'predictions_q.npy', predictions_q)

    ones = np.ones((1))


    for count, (prediction, prediction_q) in enumerate(zip(predictions,tqdm(predictions_q))):
        u, v, w = camera_intrinsics @ camera_extrinsics @ np.concatenate((prediction[:3], ones))
        img_x = u/w
        img_y = v/w

        u, v, w = camera_intrinsics @ camera_extrinsics @ np.concatenate((prediction_q[:3], ones))
        img_x_q = u/w
        img_y_q = v/w


        plt.tight_layout()
        plt.imshow(images[count][0], cmap='gray')
        plt.axis('off')
        plt.scatter(img_x, img_y, s=(0.01 + (500-0.01) * (2.5-prediction[0])))
        plt.annotate("{:.2f},  {:.2f}, {:.2f}".format(prediction[0], prediction[1], prediction[3]), (img_x, img_y))
        
        plt.scatter(img_x_q, img_y_q, s=(0.01 + (500-0.01) * (2.5-prediction_q[0])))
        plt.annotate("{:.2f},  {:.2f}, {:.2f}".format(prediction_q[0], prediction_q[1], prediction_q[3]), (img_x_q, img_y_q))
        
        plt.savefig(save_path+f"img_{count:05d}.jpg", dpi=200)
        plt.close()


if __name__=="__main__":

    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'

    img_path = '/home/pia/Crazyflie/aideck-gap8-examples/data_collection_2/raw/'

    camera_intrinsics = np.load("misc/camera_intrinsic.npy")
    camera_extrinsics = np.load("misc/camera_extrinsic.npy")

    # annotate_images_single(img_path, model_path, camera_intrinsics, camera_extrinsics, save_path='misc/annotated_img/', config=model_config, quantized=False)

    model_path_q = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.Q.pt'
    # annotate_images_single(img_path, model_path_q, camera_intrinsics, camera_extrinsics, save_path='misc/annotated_img/', config=model_config, quantized=True)

    annotate_images_both(img_path, model_path, model_path_q, camera_intrinsics, camera_extrinsics, save_path='misc/annotated_img/both/', config=model_config)