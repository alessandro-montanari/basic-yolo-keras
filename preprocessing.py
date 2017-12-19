import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, normalize, bbox_iou

def parse_annotation(ann_dir, img_dir, labels=[]):
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.counter = 0
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(len(config['ANCHORS'])//2)]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.

        self.aug_pipe = iaa.Sequential([
            sometimes((
                    # apply the following augmenters to most images
                    iaa.Fliplr(0.5), # horizontally flip 50% of all images
                    sometimes(iaa.OneOf([ 
                                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                                iaa.Affine(shear=(-20, 20)),
                                iaa.Affine(rotate=(-25, 25)),
                                iaa.Affine(translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)}),
                        ])),
                    # execute 1 to 5 of the following (less important) augmenters per image
                    # don't execute all of them, as that would often be way too strong
                    iaa.SomeOf((1, 3),
                        [
                            iaa.OneOf([
                                iaa.GaussianBlur((0.25, 2)), # blur images with a sigma between 0 and 3.0
                                iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                                iaa.MedianBlur(k=(1, 3)) # blur image using local medians with kernel sizes between 2 and 7
                            ]),
                            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
                            iaa.OneOf([
                                iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                                iaa.CoarseDropout((0.03, 0.10), size_percent=(0.30, 0.40)),
                            ]),
                            #iaa.Invert(0.05, per_channel=True), # invert color channels
                            iaa.Add((-25, 25), per_channel=0.1), # change brightness of images (by -10 to 10 of original value)
                            iaa.Multiply((0.5, 1.5), per_channel=0.1), # change brightness of images (50-150% of original value)
                            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                            iaa.PerspectiveTransform(scale=(0.025, 0.1))
                        ],
                        random_order=True
                    )
                ))],
                random_order=True
            )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))                         # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS']))                # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            # augment input image and fix object's position and size
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0, 
                                               center_w, 
                                               center_h)
                        
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                #img2 = img.copy()
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 1)
                        #cv2.putText(img2, obj['name'], 
                        #            (obj['xmin']+2, obj['ymin']+12), 
                        #            0, 1.2e-3 * img2.shape[0], 
                        #            (0,255,0), 2)
                #TODO We save the image here just to make it simple
                cv2.imwrite("/media/HD01/resizecheck" + str(idx) + ".bmp", img[:,:,::-1])

                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1  

        self.counter += 1
        #print ' new batch created', self.counter

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)
        self.counter = 0


    def aug_image(self, train_instance, jitter):
        image_name      = train_instance['filename']
        image           = cv2.imread(image_name)
        h, w, c         = image.shape
        all_objs        = copy.deepcopy(train_instance['object'])        

        bb_list     = []
        new_boxes   = []

        for obj in all_objs:      
            bb_list.append(ia.BoundingBox(x1=obj['xmin'], y1=obj['ymin'], x2=obj['xmax'], y2=obj['ymax']))
        
        bbs = ia.BoundingBoxesOnImage(bb_list , shape=image.shape)

        if jitter:
            while new_boxes==[]:
                seq_det = self.aug_pipe.to_deterministic()

                # augment image and adjust bbs
                image_aug   = seq_det.augment_image(image)   
                bbs_aug     = seq_det.augment_bounding_boxes([bbs])[0]
                bbs_aug     = bbs_aug.remove_out_of_image().cut_out_of_image()  ## removes bbs outside of pic

                # resize the image and bbs to standard size
                if(self.config['IMAGE_W']!= w and self.config['IMAGE_H']!= h):
                    image_aug = ia.imresize_single_image(image_aug, (self.config['IMAGE_W'], self.config['IMAGE_H']))
                    bbs_aug = bbs_aug.on(image_aug)

                image_aug = image_aug[:,:,::-1]

                classname   = all_objs[0]["name"]
                classindex  = all_objs[0]["class"]

                for i, bb in enumerate(bbs_aug.bounding_boxes):
                    bbwidth     = bb.x2 - bb.x1
                    bbheight    = bb.y2 - bb.y1

                    # filter bbs that are too small
                    if bbwidth > 4 and bbheight > 4:
                        new_boxes.append({
                                        'class': classindex,
                                        'name': classname,
                                        'xmin': int(bb.x1),
                                        'ymin': int(bb.y1),
                                        'xmax': int(bb.x2),
                                        'ymax': int(bb.y2),
                                        'bbwidth': int(bbwidth),
                                        'bbheight': int(bbheight)
                            })

                return image_aug, new_boxes

        else:
            # resize the image and bbs to standard size
            if(self.config['IMAGE_W']!= w and self.config['IMAGE_H']!= h):
                image = ia.imresize_single_image(image, (self.config['IMAGE_W'], self.config['IMAGE_H']))
                bbs   = bbs.on(image)

                classname   = all_objs[0]["name"]
                classindex  = all_objs[0]["class"]

                for i, bb in enumerate(bbs.bounding_boxes):
                    bbwidth     = bb.x2 - bb.x1
                    bbheight    = bb.y2 - bb.y1

                    # filter bbs that are too small
                    if bbwidth > 4 and bbheight > 4:
                        new_boxes.append({
                                        'class': classindex,
                                        'name': classname,
                                        'xmin': int(bb.x1),
                                        'ymin': int(bb.y1),
                                        'xmax': int(bb.x2),
                                        'ymax': int(bb.y2),
                                        'bbwidth': int(bbwidth),
                                        'bbheight': int(bbheight)
                            })

            else:
                new_boxes = all_objs

            image = image[:,:,::-1]

            return image, new_boxes