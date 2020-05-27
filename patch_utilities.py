import math
import torch
import numpy as np

from median_pool import MedianPool2d
from torch.nn import functional as F


class SquarePatch(torch.nn.Module):
  def __init__(self, patch_size, typ="grey", tanh=True):
    super(SquarePatch, self).__init__()
    if typ == 'grey':
      # when params are 0. the rgbs are 0.5
      self.params = torch.nn.Parameter(data=torch.full((3, patch_size, patch_size), 0, dtype=torch.float))
    elif typ == 'random':
      # uniform distribution range from -2 to -2
      self.params = torch.nn.Parameter(data=(torch.rand((3, patch_size, patch_size), dtype=torch.float) * 2 - 1) * 2)
    # both options force the patch to have valid rgb values
    if tanh:
      self.constraint = lambda params:0.5 * (torch.tanh(params) + 1)
    else:
      self.constraint = lambda params:torch.sigmoid(params)

  def forward(self):
    return self.constraint(self.params)


class SquarePatchTransformApplier(torch.nn.Module):
  def __init__(self, device:int, do_rotate=True, rand_loc=True):
    super(SquarePatchTransformApplier, self).__init__()
    self.transformer = SquarePatchTransformer().cuda(device)
    self.applier = PatchApplier().cuda(device)
    self.do_rotate = do_rotate
    self.rand_loc = rand_loc

  def forward(self, patch:torch.Tensor, batch_square_images:torch.Tensor, batch_ground_truths:torch.Tensor):
    img_size = batch_square_images.shape[2]
    transform_masks = self.transformer(patch, batch_ground_truths, img_size, self.do_rotate, self.rand_loc)
    patched_images = self.applier(batch_square_images, transform_masks)
    return patched_images


class SquarePatchTransformer(torch.nn.Module):
  """PatchTransformer: transforms batch of patches into RGB masks

  Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
  contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
  batch of labels, and pads them to the dimension of an image.

  """
  def __init__(self):
    super(SquarePatchTransformer, self).__init__()
    self.min_contrast = 0.8
    self.max_contrast = 1.2
    self.min_brightness = -0.1
    self.max_brightness = 0.1
    self.noise_factor = 0.10
    self.minangle = -20 / 180 * math.pi
    self.maxangle = 20 / 180 * math.pi
    self.medianpooler = MedianPool2d(7, same=True)

  def forward(self, adv_patch, lab_batch, img_size, do_rotate=True, rand_loc=True):
    # adv_patch = F.conv2d(adv_patch.unsqueeze(0),self.kernel,padding=(2,2))
    adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
    # Determine size of padding
    pad = (img_size - adv_patch.size(-1)) / 2
    # Make a batch of patches
    adv_patch = adv_patch.unsqueeze(0)  # .unsqueeze(0)
    # for every image label and every box in the label, make a copy of the patch
    adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1)
    batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

    # Contrast, brightness and noise transforms

    # Create random contrast tensor
    contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
    contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
    contrast = contrast.cuda()

    # Create random brightness tensor
    brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
    brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
    brightness = brightness.cuda()

    # Create random noise tensor
    noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

    # Apply contrast/brightness/noise, clamp
    adv_batch = adv_batch * contrast + brightness + noise

    adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

    # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
    cls_ids = torch.narrow(lab_batch, 2, 0, 1)
    cls_mask = cls_ids.expand(-1, -1, 3)
    cls_mask = cls_mask.unsqueeze(-1)
    cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
    cls_mask = cls_mask.unsqueeze(-1)
    cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
    msk_batch = torch.cuda.FloatTensor(cls_mask.size()).fill_(1) - cls_mask

    # Pad patch and mask to image dimensions
    mypad = torch.nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
    adv_batch = mypad(adv_batch)
    msk_batch = mypad(msk_batch)

    # Rotation and rescaling transforms
    anglesize = (lab_batch.size(0) * lab_batch.size(1))
    if do_rotate:
      angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
    else:
      angle = torch.cuda.FloatTensor(anglesize).fill_(0)

    # Resizes and rotates
    current_patch_size = adv_patch.size(-1)
    lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
    lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size
    lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size
    lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size
    lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size
    target_size = torch.sqrt(
      ((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))
    target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
    target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
    targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
    targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
    if (rand_loc):
      off_x = targetoff_x * (torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4, 0.4))
      target_x = target_x + off_x
      off_y = targetoff_y * (torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4, 0.4))
      target_y = target_y + off_y
    target_y = target_y - 0.05
    scale = target_size / current_patch_size
    scale = scale.view(anglesize)

    s = adv_batch.size()
    adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
    msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

    tx = (-target_x + 0.5) * 2
    ty = (-target_y + 0.5) * 2
    sin = torch.sin(angle)
    cos = torch.cos(angle)

    # Theta = rotation,rescale matrix
    theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = sin / scale
    theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
    theta[:, 1, 0] = -sin / scale
    theta[:, 1, 1] = cos / scale
    theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

    b_sh = adv_batch.shape
    grid = F.affine_grid(theta, adv_batch.shape, align_corners=True)

    adv_batch_t = F.grid_sample(adv_batch, grid)
    msk_batch_t = F.grid_sample(msk_batch, grid)

    adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
    msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

    adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

    return adv_batch_t * msk_batch_t


class PatchApplier(torch.nn.Module):
  """PatchApplier: applies adversarial patches to images.

  Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

  """

  def __init__(self):
    super(PatchApplier, self).__init__()

  def forward(self, img_batch, adv_batch):
    advs = torch.unbind(adv_batch, 1)
    for adv in advs:
      img_batch = torch.where((adv == 0), img_batch, adv)
    return img_batch


class TotalVariationCalculator(torch.nn.Module):
  """TotalVariation: calculates the total variation of a patch.

  Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

  """

  def __init__(self):
    super(TotalVariationCalculator, self).__init__()

  def forward(self, adv_patch):
    # bereken de total variation van de adv_patch
    tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1] + 0.000001), 0)
    tvcomp1 = torch.sum(torch.sum(tvcomp1, 0), 0)
    tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :] + 0.000001), 0)
    tvcomp2 = torch.sum(torch.sum(tvcomp2, 0), 0)
    tv = tvcomp1 + tvcomp2
    return tv / torch.numel(adv_patch)


class NPSCalculator(torch.nn.Module):
  """NMSCalculator: calculates the non-printability score of a patch.

  Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

  """

  def __init__(self, printable_rgb_file, patch_side):
    super(NPSCalculator, self).__init__()
    self.printable_rgb_array = torch.nn.Parameter(data=self._get_printable_rgb_array(printable_rgb_file, patch_side),
                                                  requires_grad=False)

  def forward(self, adv_patch):
    # calculate euclidian distance between colors in patch and colors in printability_array
    # square root of sum of squared difference
    color_dist = (adv_patch - self.printable_rgb_array + 0.000001)
    color_dist = color_dist ** 2
    color_dist = torch.sum(color_dist, 1) + 0.000001
    color_dist = torch.sqrt(color_dist)
    # only work with the min distance
    color_dist_prod = torch.min(color_dist, 0)[0]  # test: change prod for min (find distance to closest color)
    # calculate the nps by summing over all pixels
    nps_score = torch.sum(color_dist_prod, 0)
    nps_score = torch.sum(nps_score, 0)
    return nps_score / torch.numel(adv_patch)

  def _get_printable_rgb_array(self, printable_rgb_file, side):
    printability_list = []

    # read in printability triplets and put them in a list
    with open(printable_rgb_file) as f:
      for line in f:
        printability_list.append(line.split(","))

    printability_array = []
    for printability_triplet in printability_list:
      printability_imgs = []
      red, green, blue = printability_triplet
      printability_imgs.append(np.full((side, side), red))
      printability_imgs.append(np.full((side, side), green))
      printability_imgs.append(np.full((side, side), blue))
      printability_array.append(printability_imgs)

    printability_array = np.asarray(printability_array)
    printability_array = np.float32(printability_array)
    pa = torch.from_numpy(printability_array)
    return pa


class SaturationCalculator(torch.nn.Module):
    """SaturationCalculator: calculates the saturation of a patch.

    Module providing the functionality necessary to calculate the saturation of an adversarial patch.
    Instead of calculating the actual saturation per https://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
    We calculate the variance of r,g,and b scalars within a pixel.
    The more they differ from a shade of grey, (c, c, c), the higher the saturation/variance is
    These variances are averaged across all pixels to measure the saturation level of a patch
    """

    def __init__(self):
      super(SaturationCalculator, self).__init__()

    def forward(self, adv_patch):
      return torch.mean(torch.var(adv_patch, 0))

