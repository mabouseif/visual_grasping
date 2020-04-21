import math
import numpy as np
import matplotlib.pyplot as plt


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = (inp*255).astype(np.uint8)
    return inp

# def normalize_transforms(transforms, H,W):
#     transforms[0,0] = transforms[0,0]
#     transforms[0,1] = transforms[0,1]*W/H
#     transforms[0,2] = transforms[0,2]*2/H + transforms[0,0] + transforms[0,1] - 1
#
#     transforms[1,0] = transforms[1,0]*H/W
#     transforms[1,1] = transforms[1,1]
#     transforms[1,2] = transforms[1,2]*2/W + transforms[1,0] + transforms[1,1] - 1
#
#     return transforms
def normalize_transforms(transforms, W,H):
    transforms[0,0] = transforms[0,0]
    transforms[0,1] = transforms[0,1]*H/W
    transforms[0,2] = transforms[0,2]*2/W + transforms[0,0] + transforms[0,1] - 1

    transforms[1,0] = transforms[1,0]*W/H
    transforms[1,1] = transforms[1,1]
    transforms[1,2] = transforms[1,2]*2/H + transforms[1,0] + transforms[1,1] - 1

    return transforms

def rotatepoints(landmarks, center, rot):
    center_coord = np.zeros_like(landmarks)
    center_coord[:, 0] = center[0]
    center_coord[:, 1] = center[1]

    angle = math.radians(rot)

    rot_matrix = np.array([[math.cos(angle), -1 * math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])

    rotated_coords = np.dot((landmarks - center_coord), rot_matrix) + center_coord

    return rotated_coords

def show_image(image, gray=False):
    # print(image.shape)
    fig = plt.figure(figsize=plt.figaspect(.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(image)
    if gray:
        ax.imshow(image, cmap='gray')
    ax.axis('off')
    plt.show()