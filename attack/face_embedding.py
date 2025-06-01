import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
import torch

import numpy as np
from numpy.linalg import norm as l2norm

class Face(dict):

    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        #for k in self.__class__.__dict__.keys():
        #    if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
        #        setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                    if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(Face, self).__setattr__(name, value)
        super(Face, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __getattr__(self, name):
        return None

    @property
    def embedding_norm(self):
        if self.embedding is None:
            return None
        return l2norm(self.embedding)

    @property 
    def normed_embedding(self):
        if self.embedding is None:
            return None
        return self.embedding / self.embedding_norm

    @property 
    def sex(self):
        if self.gender is None:
            return None
        return 'M' if self.gender==1 else 'F'


class FaceEmbeddingExtractor:
    def __init__(self, model_name='antelopev2', root='../'):
        """
        Initialize the face embedding extractor with InsightFace's ArcFace model.
        
        Args:
            model_name (str): Name of the model to use (default: 'antelopev2')
            root (str): Root directory for model files
        """
        self.arcface_model = FaceAnalysis(
            name=model_name,
            root=root,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.arcface_model.prepare(ctx_id=0, det_size=(640, 640))

    def extract_embedding(self, image):
        """
        Extract face embedding from an RGB image.
        
        Args:
            image: RGB image as numpy array or PIL Image
            
        Returns:
            numpy.ndarray: Face embedding vector if face is detected, None otherwise
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image) # (512, 512, 3)
            
        # # Convert RGB to BGR for InsightFace
        # if len(image.shape) == 3 and image.shape[2] == 3:
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 测试，先将image转换为tensor的rgb，然后手动转换为tensor的bgr
        numpy_image = image.copy()
        image = torch.Tensor(np.transpose(image, (2, 0, 1))) # torch.Size([3, 512, 512])
        
        # # Get face information
        # face_info = self.arcface_model.get(image)
        import pdb; pdb.set_trace()
        # 只需要recognition模型
        # 下面是get函数的内部，首先检测人脸并返回bbox和关键点
        bboxes, kpss = self.arcface_model.det_model.detect(image, # (1280, 1280, 3)
                                             max_num=0, # 0
                                             metric='default') # (1, 5), (1, 5, 2)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]): # 1
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None   
            if kpss is not None:
                kps = kpss[i] # (5, 2)
            face = Face(bbox=bbox, kps=kps, det_score=det_score) # dict_keys(['bbox', 'kps', 'det_score'])
            for taskname, model in self.arcface_model.models.items(): # dict_items([('landmark_3d_68', <insightface.model_zoo.landmark.Landmark object at 0x73105d530dc0>), ('landmark_2d_106', <insightface.model_zoo.landmark.Landmark object at 0x73105d531c60>), ('genderage', <insightface.model_zoo.attribute.Attribute object at 0x73105d531ab0>), ('recognition', <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x73105d5329e0>), ('detection', <insightface.model_zoo.retinaface.RetinaFace object at 0x73105d530520>)])
                # if taskname=='detection':
                #     continue
                if taskname != 'recognition':
                    continue
                # 用到的是'recognition', <insightface.model_zoo.arcface_onnx.ArcFaceONNX object at 0x73105d5329e0>
                model.get(image, face)
                
            ret.append(face) # face.keys() dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
        
        face_info = ret
        
        if not face_info:
            return None
            
        # Sort faces by size and get the largest one
        face_info = sorted(
            face_info,
            key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1])
        )[-1]
        
        # Return the face embedding
        return face_info['embedding']

def main():
    # Example usage
    extractor = FaceEmbeddingExtractor()
    import pdb; pdb.set_trace()
    # Load and process an image
    image_path = "/data1/humw/Codes/FaceOff/datasets/test/n000050/set_B/0012_01.png"
    image = Image.open(image_path).convert('RGB')
    
    # Extract embedding
    embedding = extractor.extract_embedding(image)
    
    if embedding is not None:
        print(f"Face embedding shape: {embedding.shape}")
        print(f"Face embedding: {embedding}")
    else:
        print("No face detected in the image")

if __name__ == "__main__":
    main()