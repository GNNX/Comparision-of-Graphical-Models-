import cv2, torch
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.utils import cv2_util
import numpy as np
from time import time
from torchnet import meter

print('ABC')

class COCODemo(object):
    CATEGORIES = [
     '__background',
     'person',
     'bicycle',
     'car',
     'motorcycle',
     'airplane',
     'bus',
     'train',
     'truck',
     'boat',
     'traffic light',
     'fire hydrant',
     'stop sign',
     'parking meter',
     'bench',
     'bird',
     'cat',
     'dog',
     'horse',
     'sheep',
     'cow',
     'elephant',
     'bear',
     'zebra',
     'giraffe',
     'backpack',
     'umbrella',
     'handbag',
     'tie',
     'suitcase',
     'frisbee',
     'skis',
     'snowboard',
     'sports ball',
     'kite',
     'baseball bat',
     'baseball glove',
     'skateboard',
     'surfboard',
     'tennis racket',
     'bottle',
     'wine glass',
     'cup',
     'fork',
     'knife',
     'spoon',
     'bowl',
     'banana',
     'apple',
     'sandwich',
     'orange',
     'broccoli',
     'carrot',
     'hot dog',
     'pizza',
     'donut',
     'cake',
     'chair',
     'couch',
     'potted plant',
     'bed',
     'dining table',
     'toilet',
     'tv',
     'laptop',
     'mouse',
     'remote',
     'keyboard',
     'cell phone',
     'microwave',
     'oven',
     'toaster',
     'sink',
     'refrigerator',
     'book',
     'clock',
     'vase',
     'scissors',
     'teddy bear',
     'hair drier',
     'toothbrush']

    def __init__(self, cfg, confidence_threshold=0.7, show_mask_heatmaps=False, masks_per_dim=2, min_image_size=224):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size
        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)
        self.transforms = self.build_transform()
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([33554431, 32767, 2097151])
        self.cpu_device = torch.device('cpu')
        self.confidence_threshold = confidence_threshold
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])
        normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        transform = T.Compose([
         T.ToPILImage(),
         T.Resize(self.min_image_size),
         T.ToTensor(),
         to_bgr_transform,
         normalize_transform])
        return transform

    def run_on_opencv_image(self, image, predictions):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        top_predictions = self.select_top_predictions(predictions)
        result = image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        if self.cfg.MODEL.KEYPOINT_ON:
            result = self.overlay_keypoints(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)
        return result

    def compute_prediction(self, original_image):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]
        prediction = predictions[0]
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))
        if prediction.has_field('mask'):
            masks = prediction.get_field('mask')
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field('mask', masks)
        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field('scores')
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field('scores')
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype('uint8')
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field('labels')
        boxes = predictions.bbox
        colors = self.compute_colors_for_labels(labels).tolist()
        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 1)

        return image

    def overlay_mask(self, image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.
        
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field('mask').numpy()
        labels = predictions.get_field('labels')
        colors = self.compute_colors_for_labels(labels).tolist()
        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image
        return composite

    def overlay_keypoints(self, image, predictions):
        keypoints = predictions.get_field('keypoints')
        kps = keypoints.keypoints
        scores = keypoints.get_field('logits')
        kps = torch.cat((kps[:, :, 0:2], scores[:, :, None]), dim=2).numpy()
        for region in kps:
            image = vis_keypoints(image, region.transpose((1, 0)))

        return image

    def create_mask_montage(self, image, predictions):
        """
        Create a montage showing the probability heatmaps for each one one of the
        detected objects
        
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask`.
        """
        masks = predictions.get_field('mask')
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate(masks.float(), scale_factor=1 / masks_per_dim).byte()
        height, width = masks.shape[-2:]
        max_masks = masks_per_dim ** 2
        masks = masks[:max_masks]
        if len(masks) < max_masks:
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8)
            masks_padded[:len(masks)] = masks
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width)
        result = torch.zeros((
         masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8)
        for y in range(masks_per_dim):
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[(y, x)]

        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET)

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field('scores').tolist()
        labels = predictions.get_field('labels').tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox
        template = '{}: {:.2f}'
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255,
                                                                          255), 1)

        return image


import numpy as np, matplotlib.pyplot as plt
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints

def vis_keypoints(img, kps, kp_thresh=2, alpha=0.7):
    """Visualizes keypoints (adapted from vis_one_image).
    kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
    """
    dataset_keypoints = PersonKeypoints.NAMES
    kp_lines = PersonKeypoints.CONNECTIONS
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    kp_mask = np.copy(img)
    mid_shoulder = (kps[:2, dataset_keypoints.index('right_shoulder')] + kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
    sc_mid_shoulder = np.minimum(kps[(2, dataset_keypoints.index('right_shoulder'))], kps[(2, dataset_keypoints.index('left_shoulder'))])
    mid_hip = (kps[:2, dataset_keypoints.index('right_hip')] + kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
    sc_mid_hip = np.minimum(kps[(2, dataset_keypoints.index('right_hip'))], kps[(2, dataset_keypoints.index('left_hip'))])
    nose_idx = dataset_keypoints.index('nose')
    if sc_mid_shoulder > kp_thresh and kps[(2, nose_idx)] > kp_thresh:
        cv2.line(kp_mask, tuple(mid_shoulder), tuple(kps[:2, nose_idx]), color=colors[len(kp_lines)], thickness=2, lineType=cv2.LINE_AA)
    if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
        cv2.line(kp_mask, tuple(mid_shoulder), tuple(mid_hip), color=colors[(len(kp_lines) + 1)], thickness=2, lineType=cv2.LINE_AA)
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = (kps[(0, i1)], kps[(1, i1)])
        p2 = (kps[(0, i2)], kps[(1, i2)])
        if kps[(2, i1)] > kp_thresh and kps[(2, i2)] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[(2, i1)] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[(2, i2)] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)


def make_x(labels, scores):
    ret = torch.zeros(81)
    m = torch.ones_like(ret)
    m[53] = 0
    m[79] = 0
    label_set = list(set(list(labels.detach().cpu().numpy())))
    for i in label_set:
        mask = torch.where(labels == i, torch.ones_like(labels), torch.zeros_like(labels)).float()
        ret[i] = torch.max(mask * scores)

    return ret * m

def change(x_s, adj):
    #x_s = x_s.t()
    pad = torch.zeros(81, 9).cuda()
    x_s = torch.cat([x_s, pad], dim = 1)[None, ::]
    #print (x_s.shape, pad.shape)
    
    #adj = torch.ones(81,81,1).cuda()
    adj = adj[:,:,None]
    #print (x_s.shape, adj.shape)
    return x_s, adj

from GCN_models import GCN
import random as rd

def train_gcn(coco_demo, adj, map_imgid2y, map_imgid2x=None):
    model = GCN(1, 5, 5, dropout=0.5).cuda()
    adj = adj.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
    Loss = torch.nn.SmoothL1Loss()
    print('Starting Training')
    minloss = 1000
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    st = time()
    batch_size = 120
    for epoch in range(10):
        for n_iter in range(500):
            batch = rd.sample([i for i in range(60000)], batch_size)
            loss = 0
            i = -1
            k = 0
            for _ in batch:
                imgname = imglist[_]
                if map_imgid2x is not None:
                    x_s = map_imgid2x[imgname[:-4]][:, None].cuda()
                else:
                    predictions = coco_demo.compute_prediction(cv2.imread('VG_100K/' + imgname))
                    x_s = make_x(predictions.get_field('labels'), predictions.get_field('scores'))[:, None].cuda()
                y_s = map_imgid2y[imgname[:-4]].cuda()
                i += 1
                o = torch.squeeze(model(x_s, adj))
                loss += Loss(o, y_s)
                z = 1
                if i % z == 0 and i != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    fps = z / (time() - st)
                    print('[EPOCH]:' + str(epoch) + ' [ITER]:' + str(n_iter) + ' [BATCH]:' + str(k) + ' [MINLOSS]:' + str(minloss / z) + ' [LOSS]:' + str(loss.item() / z))
                    k += 1
                    st = time()
                    if loss.item() < minloss:
                        print('saving')
                        torch.save(model.state_dict(), 'models_GCN_SL1/GCN_1_5_5_1_SL1_' + str(loss.item()) + '.pt')
                        minloss = loss.item()
                    loss = 0


from GAT_models import GAT
from GAT_models import SpGAT

def train_GAT(coco_demo, adj, map_imgid2y, map_imgid2x=None):
    model = GAT(1, 5, 1, 0.5, 0.02, 5).cuda()
    adj = adj.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-05, weight_decay=0.0005)
    Loss = torch.nn.BCEWithLogitsLoss()
    print('Starting Training')
    minloss = 1000
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    batch_size = 120
    for epoch in range(10):
        for n_iter in range(500):
            batch = rd.sample([i for i in range(60000)], batch_size)
            loss = 0
            i = -1
            k = 0
            for _ in batch:
                imgname = imglist[_]
                if map_imgid2x is not None:
                    x_s = map_imgid2x[imgname[:-4]][:, None].cuda()
                else:
                    predictions = coco_demo.compute_prediction(cv2.imread('VG_100K/' + imgname))
                    x_s = make_x(predictions.get_field('labels'), predictions.get_field('scores'))[:, None].cuda()
                y_s = map_imgid2y[imgname[:-4]].cuda()
                i += 1
                o = torch.squeeze(model(x_s, adj))
                print(torch.sum(o), torch.sum(y_s))
                loss += Loss(o, y_s)
                z = 1
                if i % z == 0 and i != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('[EPOCH]:' + str(epoch) + ' [ITER]:' + str(n_iter) + ' [BATCH]:' + str(k) + ' [LOSS]:' + str(loss.item() / z))
                    k += 1
                    if loss.item() < minloss:
                        print('saving')
                        torch.save(model.state_dict(), 'models_GAT_BCELL/GAT_1_5_1_5_BCELL_' + str(loss.item()) + '.pt')
                        minloss = loss.item()
                    loss = 0


from GGNN_models import GGNN
import random as rd

def train_GGNN(adj, map_imgid2y, map_imgid2x=None):
    model = GGNN(81, 10, 81, 1).cuda()
    adj = adj.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)
    Loss = torch.nn.MSELoss()
    print('Starting Training')
    minloss = 1000
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    st = time()
    batch_size = 120
    for epoch in range(10):
        for n_iter in range(500):
            batch = rd.sample([i for i in range(60000)], batch_size)
            loss = 0
            i = -1
            k = 0
            for _ in batch:
                imgname = imglist[_]
                if map_imgid2x is not None:
                    x_s = map_imgid2x[imgname[:-4]][:, None].cuda()
                else:
                    predictions = coco_demo.compute_prediction(cv2.imread('VG_100K/' + imgname))
                    x_s = make_x(predictions.get_field('labels'), predictions.get_field('scores'))[:, None].cuda()
                
                y_s = map_imgid2y[imgname[:-4]].cuda()
                i += 1
                
                x_s, adj1 = change(x_s, adj)

                o = torch.squeeze(model(x_s, adj1, adj1))
                loss += Loss(o, y_s)
                z = 1
                if i % z == 0 and i != 0:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    fps = z / (time() - st)
                    print('[EPOCH]:' + str(epoch) + ' [ITER]:' + str(n_iter) + ' [BATCH]:' + str(k) + ' [MINLOSS]:' + str(minloss / z) + ' [LOSS]:' + str(loss.item() / z))
                    k += 1
                    st = time()
                    if loss.item() < minloss:
                        print('saving')
                        torch.save(model.state_dict(), 'models_GGNN_MSE/GGNN_3_MSE_' + str(loss.item()) + '.pt')
                        minloss = loss.item()
                    loss = 0

def train(epoch, dataloader, net, criterion, optimizer, opt):
    net.train()
    for i, (adj_matrix, annotation, target) in enumerate(dataloader, 0):
        net.zero_grad()

        padding = torch.zeros(len(annotation), opt.n_node, opt.state_dim - opt.annotation_dim).double()
        init_input = torch.cat((annotation, padding), 2)
        
        init_input = init_input.cuda()
        adj_matrix = adj_matrix.cuda()
        annotation = annotation.cuda()
        target = target.cuda()

        output = net(init_input, annotation, adj_matrix)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        if i % int(len(dataloader) / 10 + 1) == 0 and opt.verbal:
            print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, opt.niter, i, len(dataloader), loss.data[0]))

def demonstrate(coco_demo, grph_model, adj, N=5):
    if grph_model[9:12] == 'GCN':
        model = GCN(1, 5, 5, dropout=0.5).cuda()
    if grph_model[9:12] == 'GAT':
        model = GAT(1, 5, 1, 0.5, 0.02, 5).cuda()
    model.load_state_dict(torch.load(grph_model))
    model.eval()
    adj = adj.cuda()
    imglist = []
    with open('dvimages.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    sampled_imgs = rd.sample(imglist, N)
    for imgname in sampled_imgs:
        img = cv2.imread('VG_100K/' + imgname)
        predictions = coco_demo.compute_prediction(img)
        x_s = make_x(predictions.get_field('labels'), predictions.get_field('scores'))[:, None].cuda()
        y_s = map_imgid2y[imgname[:-4]]
        o = model(x_s, adj)
        o = o.detach().cpu()
        o = torch.squeeze(o)
        o = o / torch.max(o)
        o = y_s - o
        o[o < 0] = 0
        o[o > 0] = 1
        output_classes = [i for i in range(len(o)) if o[i] == 1.0]
        t_predictions = coco_demo.select_top_predictions(predictions)
        p_labels = [i for i in list(predictions.get_field('labels').numpy())]
        top_labels = [i for i in list(t_predictions.get_field('labels').numpy())]
        d_viola = []
        for cls in output_classes:
            if cls in p_labels and cls not in top_labels:
                d_viola.append(cls)

        vis_img = coco_demo.run_on_opencv_image(img, predictions)
        cv2.imwrite(imgname + '_pre' + '.png', vis_img)
        for obj in d_viola:
            m1 = (predictions.get_field('labels') == obj).float()
            m2 = (predictions.get_field('scores') == torch.max(predictions.get_field('scores') * m1)).float()
            m3 = predictions.get_field('scores') * m2
            m3[m3 != 0] = 1 - torch.max(m3)
            set_to = predictions.get_field('scores') + m3
            for i in range(len(set_to)):
                predictions.get_field('scores')[i] = set_to[i]

        vis_img = coco_demo.run_on_opencv_image(img, predictions)
        cv2.imwrite(imgname + '_post' + '.png', vis_img)


def print_cwise(tens):
    CATEGORIES = [
     '__background',
     'person',
     'bicycle',
     'car',
     'motorcycle',
     'airplane',
     'bus',
     'train',
     'truck',
     'boat',
     'traffic light',
     'fire hydrant',
     'stop sign',
     'parking meter',
     'bench',
     'bird',
     'cat',
     'dog',
     'horse',
     'sheep',
     'cow',
     'elephant',
     'bear',
     'zebra',
     'giraffe',
     'backpack',
     'umbrella',
     'handbag',
     'tie',
     'suitcase',
     'frisbee',
     'skis',
     'snowboard',
     'sports ball',
     'kite',
     'baseball bat',
     'baseball glove',
     'skateboard',
     'surfboard',
     'tennis racket',
     'bottle',
     'wine glass',
     'cup',
     'fork',
     'knife',
     'spoon',
     'bowl',
     'banana',
     'apple',
     'sandwich',
     'orange',
     'broccoli',
     'carrot',
     'hot dog',
     'pizza',
     'donut',
     'cake',
     'chair',
     'couch',
     'potted plant',
     'bed',
     'dining table',
     'toilet',
     'tv',
     'laptop',
     'mouse',
     'remote',
     'keyboard',
     'cell phone',
     'microwave',
     'oven',
     'toaster',
     'sink',
     'refrigerator',
     'book',
     'clock',
     'vase',
     'scissors',
     'teddy bear',
     'hair drier',
     'toothbrush']
    for i in range(81):
        print(CATEGORIES[i], tens[i])

def thresh(y_s, o, md):
    o = o.detach().cpu().squeeze()
    if md == 1:
      
        o = o / torch.max(o)
        o = y_s - o
        
        #print (o)
        o[o > 0] = 1
        o[o < 0] = 0

        #print ('y',y_s)
        #print ('o',o)
        
        return o
 
    if md == 2:
        o = o/torch.min(o)
        o[o > 10] = 1
        return o
    if md == 3:
        o = o/torch.max(o)
        o[o >= 0.5] = 1
        o[o < 0.5] = 0
        return o

def save_output(grph_model, map_imgid2x, map_imgid2y, adj):
    if 'GCN' in grph_model:
        model = GCN(1, 5, 5, dropout=0.5).cuda()
    if 'GAT' in grph_model:
        model = GAT(1, 5, 1, 0.5, 0.02, 5).cuda()
    if 'GGNN' in grph_model:
        model = GGNN(81, 10, 81, 1).cuda()
    
    model.load_state_dict(torch.load(grph_model))
    model.eval()
    adj = adj.cuda()
    print('Starting Testing')
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    img_test = imglist[60000:]
    output = []
    k = 0
    s = [0,0]
    for imgname in img_test:
        if k % 100 == 0:
            print(k)
            print (s[0],s[1])
            s = [0,0]
        k += 1
        x_s = map_imgid2x[imgname[:-4]][:, None].cuda()
        y_s = map_imgid2y[imgname[:-4]]
        
        if 'GGNN' in grph_model:
            x_s, adj1 = change(x_s, adj)
            o = model(x_s, adj1, adj1).detach().cpu().squeeze()
            #o = thresh(y_s, o2, 3)

        else:
            o = model(x_s, adj).detach().cpu().squeeze()
        #s[0]+=(torch.sum(y_s - o1).item())
        #s[1]+=(torch.sum(y_s - o2).item())
        o = o.numpy()
 
        #print (y_s, o1)

        
        output.append(o)

    with open('output_GAT_MSE_wot.txt', 'w') as (f):
        for i in range(len(output)):
            f.write(img_test[i][:-4] + ' ' + ('').join(str(output[i].tolist())) + '\n')


def save_input(coco_demo):
    print('Starting Saving')
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    img_test = imglist[:60000]
    input_ = []
    k = 0
    for imgname in img_test:
        if k % 10 == 0:
            print(k)
        k += 1
        img = cv2.imread('VG_100K/' + imgname)
        predictions = coco_demo.compute_prediction(img)
        x_s = make_x(predictions.get_field('labels'), predictions.get_field('scores'))[:, None].cuda()
        x_s = x_s.detach().cpu()
        x_s = torch.squeeze(x_s)
        x_s = x_s.numpy()
        input_.append(x_s)

    with open('input_coco_train.txt', 'w') as (f):
        for i in range(len(input_)):
            f.write(img_test[i][:-4] + ' ' + ('').join(str(input_[i].tolist())) + '\n')


def evaluate(map_imgid2y, map_imgid2o, lvl_sel, N=50, S=1000):
    print('Starting Evaluation')
    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    acc = torch.zeros(81, N).cuda()
    prec = torch.zeros(81, N).cuda()
    recal = torch.zeros(81, N).cuda()
    for j in range(N):
        picked_imgs = rd.sample([i for i in range(60000, len(map_imgid2y))], S)
        y_pile = []
        out_pile = []
        for i in picked_imgs:
            imgname = imglist[i]
            y_s = map_imgid2y[imgname[:-4]]
            y_pile.append(y_s)
            t_o = map_imgid2o[imgname[:-4]]
            out_pile.append(t_o)

        y_pile = torch.stack(y_pile).t().cuda()
        out_pile = torch.stack(out_pile).t().cuda()
        for i in range(81):
            s = y_pile[i] + out_pile[i]
            d1 = y_pile[i] - out_pile[i]
            d2 = out_pile[i] - y_pile[i]
            tp = torch.sum((s == 2).float())
            tn = torch.sum((s == 0).float())
            fp = torch.sum((d1 == -1).float())
            fn = torch.sum((d2 == -1).float())
            acc[i, j] = (tp + tn) / (tp + tn + fp + fn)
            prec[i, j] = tp / (tp + fp + 1e-05)
            recal[i, j] = tp / (tp + fn + 1e-05)
            #print (prec[i, j], recal[i, j])

    print('Overall Acc:', torch.mean(acc))
    print('Overall Preccision:', torch.mean(prec))
    print('Overall Recall:', torch.mean(recal))
    recal_lvls = []
    level = []
    q = 0
    while True:
        level.append(q)
        q += 1 / lvl_sel
        if q > 1:
            break
    

    for i in range(len(level)-1):
        m1 = (recal >= level[i]).float()
        m2 = (recal <= level[i+1]).float()
        mask = (m1 + m2 == 2.0).float()
        #print (torch.sum(mask))
        t_prec = prec * mask
        recal_lvls.append(torch.mean(t_prec, dim=1))

    recal_lvls = torch.stack(recal_lvls).t()
    #print (recal_lvls)
    mask = torch.sum((recal_lvls > 0.).float(), dim = 1)
    mask += (mask == 0.).float()*1e-05
    
    mAP = torch.sum(recal_lvls, dim=1)/mask
    #print (mAP)
    print('Overall mAP:', torch.mean(mAP))


def load_tensors(f_name):
    ret = []
    imgid = []
    with open(f_name) as (f):
        while True:
            s = f.readline()
            if s == '':
                break
            imgidi = s.strip('\n').split()[0]
            ri = s[len(imgidi) + 1:].strip(' []\n')
            imgid.append(imgidi)
            ret.append(torch.FloatTensor(list(map(float, ri.split(',')))))

    map_ = dict(zip(imgid, ret))
    return map_



def evaluate1(map_imgid2y, map_imgid2o, N=50, S=1000):
    print('Starting Evaluation')
    mAPM = meter.mAPMeter()
    APM = meter.APMeter()

    imglist = []
    with open('images_coco.txt') as (f):
        while True:
            imgname = f.readline().strip('\n')
            if imgname == '':
                break
            imglist.append(imgname)

    for j in range(N):
        picked_imgs = rd.sample([i for i in range(60000, len(map_imgid2y))], S)
        y_pile = []
        out_pile = []
        for i in picked_imgs:
            imgname = imglist[i]
            y_s = map_imgid2y[imgname[:-4]]
            y_pile.append(y_s)
            t_o = map_imgid2o[imgname[:-4]]
            out_pile.append(t_o)

            
        y_pile = torch.stack(y_pile).cuda()
        out_pile = torch.stack(out_pile).cuda()
        mAPM.add(out_pile, y_pile)
        APM.add(out_pile, y_pile)

        


    print('Overall mAP:', 100*mAPM.value())
    print('Classwise AP:', 100*APM.value())
    


    # recal_lvls = []
    # level = []
    # q = 0
    # while True:
    #     level.append(q)
    #     q += 1 / lvl_sel
    #     if q > 1:
    #         break
    

    # for i in range(len(level)-1):
    #     m1 = (recal >= level[i]).float()
    #     m2 = (recal <= level[i+1]).float()
    #     mask = (m1 + m2 == 2.0).float()
    #     #print (torch.sum(mask))
    #     t_prec = prec * mask
    #     recal_lvls.append(torch.mean(t_prec, dim=1))

    # recal_lvls = torch.stack(recal_lvls).t()
    # #print (recal_lvls)
    # mask = torch.sum((recal_lvls > 0.).float(), dim = 1)
    # mask += (mask == 0.).float()*1e-05
    
    # mAP = torch.sum(recal_lvls, dim=1)/mask
    # #print (mAP)
    # print('Overall mAP:', torch.mean(mAP))



# from maskrcnn_benchmark.config import cfg
# from predictor import COCODemo
# import os, pickle
# config_file = '../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml'
# cfg.merge_from_file(config_file)
# cfg.merge_from_list(['MODEL.DEVICE', 'cuda'])
# print('Loading Visual Model')
# coco_demo = COCODemo(cfg, min_image_size=800, confidence_threshold=0.7)

print('Loading Adj Matrix')
adj = torch.zeros(81, 81)
i = 0
with open('adj_coco_2D_renorm.txt') as (f):
    while True:
        adji = f.readline().strip('[]\n')
        if adji == '':
            break
        adj[i] = torch.FloatTensor(list(map(float, adji.split(','))))
        i += 1

print('Loading All the Labels')
map_imgid2y = load_tensors('y_coco_full.txt')
print('Loading All the Outputs')
map_imgid2oGCN = load_tensors('output_GGNN_SL1_wot.txt')
#map_imgid2oGAT = load_tensors('output_GAT.txt')
#map_imgid2oGGNN = load_tensors('output_GGNN_BCE.txt')
print('Loading All the Inputs')
map_imgid2x = load_tensors('input_coco_test.txt')


GCN_model = 'models_GCN_SL1/GCN_1_5_5_1_SL1_0.0007358308066613972.pt'
GAT_model = 'models_GAT_MSE/GAT_1_5_1_5_MSE_0.0023521368857473135.pt'
GGNN_model = 'models_GGNN_SL1/GGNN_3_SL1_0.00016544421669095755.pt'


#train_GGNN(adj, map_imgid2y, map_imgid2x)
save_output(GAT_model, map_imgid2x, map_imgid2y, adj)
#train_GAT(coco_demo, adj, map_imgid2y, map_imgid2x=map_imgid2x)
#evaluate1(map_imgid2y, map_imgid2oGCN, N=50, S=1000)
#demonstrate()
exit()

