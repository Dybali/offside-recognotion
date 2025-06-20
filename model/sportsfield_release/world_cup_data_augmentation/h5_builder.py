import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append("..")

import numpy as np
import tables

import raw_data_loader


from ..utils import utils

class WorldCupH5Builder():
    def __init__(self, data_dir, dataset_type):
        self.total_samples = 50000
        self.loader = raw_data_loader.RawDataloader(dataset_type)
        self.file_name = 'world_cup_{0}.h5'.format(dataset_type)
        self.samples_per_image = 1
        
        if dataset_type == 'test':
            self.id_range = range(1, 186+1)
        elif dataset_type == 'train':
            self.id_range = range(1, 170+1)  # Train için 1'den 170'e kadar
        elif dataset_type == 'val':
            self.id_range = range(171, 209+1)   # Validation için 171'den 209'a kadar
        else:
            raise NotImplementedError()

    def init_h5_file(self):
        h5file = tables.open_file(self.file_name, mode="w", title="worldcup dataset")
        filters = tables.Filters(complevel=5, complib='blosc')
        video_storage = h5file.create_earray(
            h5file.root,
            'frames',
            tables.Atom.from_dtype(np.dtype(np.uint8)),
            shape=(0, 256, 256, 3),
            filters=filters,
            expectedrows=10000000)
        homography_storage = h5file.create_earray(
            h5file.root,
            'homographies',
            tables.Atom.from_dtype(np.dtype(np.float64)),
            shape=(0, 3, 3),
            filters=filters,
            expectedrows=10000000)
        storage = [h5file, video_storage, homography_storage]
        return storage

    def append_data(self, storage):
        h5file, video_storage, homography_storage = storage
        for image_id in self.id_range: 
            for _ in range(self.samples_per_image):
                cropped_frame, cropped_homography = self.loader.get_paired_data_by_id(image_id)
                cropped_frame = cropped_frame * 255.0
                cropped_frame = cropped_frame.astype(np.uint8)
                homography_storage.append(cropped_homography[None])
                video_storage.append(cropped_frame[None])
        h5file.close()

    def build_h5(self):
        storage = self.init_h5_file()
        self.append_data(storage)


def main():
    # Test veri seti oluştur
    test_builder = WorldCupH5Builder('test', 'test')
    test_builder.build_h5()
    
    # Train veri seti oluştur
    train_builder = WorldCupH5Builder('train', 'train')
    train_builder.build_h5()
    
    # Validation veri seti oluştur
    val_builder = WorldCupH5Builder('val', 'val')
    val_builder.build_h5()


if __name__ == '__main__':
    main()
