import os.path as op
from typing import List
import sys
import json
sys.path.append("..")

from utils.iotools import read_json
from .bases import BaseDataset


class RSTPReid(BaseDataset):
    """
    RSTPReid

    Reference:
    DSSL: Deep Surroundings-person Separation Learning for Text-based Person Retrieval MM 21

    URL: http://arxiv.org/abs/2109.05534

    Dataset statistics:
    # identities: 4101 
    """
    dataset_dir = 'RSTPReid'

    def __init__(self, root='', verbose=True, part_seg=False):
        super(RSTPReid, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.anno_path = op.join(self.dataset_dir, 'data_captions.json')
        self._check_before_run()

        self.train_annos, self.test_annos, self.val_annos = self._split_anno(self.anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        if part_seg:
            self.seg_img_dir = op.join(self.dataset_dir, 'segs/')
            img_path_dict = {item[-2].split("imgs/")[-1]:idx for idx,item in enumerate(self.train)}

            # read processed attribute data
            # 将所有的文本进行分割后得到的新数据json，进行读取
            with open(op.join(self.dataset_dir, "RSTPReid_data_final_37010.json"),"r+") as f:
                data = json.load(f)
            
            # 读取使用SAM分割后的所有数据集名称以及置信度
            with open(op.join(self.dataset_dir, "RSTPReid_score_dict_37009.json"),"r+") as f:
                data_seg = json.load(f)

            train = self.train
            train_data = []
            # 处理每一条json，并设置进数据集中
            for data_id,item in enumerate(data):
                #item
                idx = item['idx']
                origin_data = train[idx]
                # 分割结果的存放
                seg_img_name = ["_".join([str(idx)] + item['name'][:-len(".png")].split("/")+[str(i)]) for i in range(len(item['attribute']))]
                seg_img_score = [data_seg[k] for k in seg_img_name] # 分割后的置信度
                attribute = item['attribute'] # 分割后的文本属性数据
                if isinstance(attribute[0], list):
                    continue
                seg_img_name = [op.join(self.seg_img_dir,k+".png") for k in seg_img_name]
                origin_data = origin_data + (attribute,seg_img_name,seg_img_score)
                train_data.append(origin_data)
                assert self.train[idx][-2].split("imgs")[-1] == item['name']
            
            self.train = train_data
            pass
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSTPReid Images and Captions are loaded")
            self.show_dataset_info()


    def _split_anno(self, anno_path: str):
        train_annos, test_annos, val_annos = [], [], []
        annos = read_json(anno_path)
        for anno in annos:
            if anno['split'] == 'train':
                train_annos.append(anno)
            elif anno['split'] == 'test':
                test_annos.append(anno)
            else:
                val_annos.append(anno)
        return train_annos, test_annos, val_annos

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                captions = anno['captions'] # caption list
                for caption in captions:
                    dataset.append((pid, image_id, img_path, caption))
                image_id += 1
            for idx, pid in enumerate(pid_container):
                # check pid begin from 0 and no break
                assert idx == pid, f"idx: {idx} and pid: {pid} are not match"
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            for anno in annos:
                pid = int(anno['id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['img_path'])
                img_paths.append(img_path)
                image_pids.append(pid)
                caption_list = anno['captions'] # caption list
                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(pid)
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.anno_path):
            raise RuntimeError("'{}' is not available".format(self.anno_path))
