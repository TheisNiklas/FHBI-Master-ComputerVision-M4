import tensorflow as tf
import os
import pandas as pd


class DataLoader:
    # parameters
    imageHeight: int
    imageWidth: int
    trainSplit: float
    valSplit: float
    testSplit: float
    shuffle: bool
    seed: int

    def __init__(
        self,
        imageHeight: int = 224,
        imageWidth: int = 224,
        trainSplit: float = 0.70,
        valSplit: float = 0.15,
        testSplit: float = 0.15,
        shuffle: bool = True,
        seed: int = 123,
    ):
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.trainSplit = trainSplit
        self.valSplit = valSplit
        self.testSplit = testSplit
        self.shuffle = shuffle
        self.seed = seed

    def loadDatasets(
        self, relDataDir: str, preprocessedDataName: str , batchSize: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Loads a dataset and split it into train, validation and test datasets

        Args:
            relDataDir (str): relative to the "data"-folder
            batchSize (int): batch size for the datasets

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: datasets in the order train, validation, test
        """
        metaDataFile = os.path.join("../data_meta", relDataDir, "processed", preprocessedDataName)
        metaData = pd.read_json(metaDataFile)
        anchor_images = metaData["anchor"]
        positive_images = metaData["positive"]
        negative_images = metaData["negative"]
        
        anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
        positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)
        negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)
        
        dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
        dataset = dataset.map(DataLoader.preprocess_triplets)
        
        dsSize = dataset.__len__().numpy()
        return self.__createDatasets(dataset, dsSize, batchSize)
    
    def loadDatasetsPairs(
        self, relDataDir: str, preprocessedDataName: str , batchSize: int, crop: bool = True
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Loads a dataset and split it into train, validation and test datasets

        Args:
            relDataDir (str): relative to the "data"-folder
            batchSize (int): batch size for the datasets

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: datasets in the order train, validation, test
        """
        metaDataFile = os.path.join("../data_meta", relDataDir, "processed", preprocessedDataName)
        metaData = pd.read_json(metaDataFile)
        anchor_images = metaData["anchor"]
        compare_images = metaData["compare"]
        labels = metaData["label"]
        
        dataset = tf.data.Dataset.from_tensor_slices((anchor_images, compare_images, labels))
        if crop:
            dataset = dataset.map(DataLoader.decode_imgs)
        else:
            dataset = dataset.map(DataLoader.decode_imgs_no_crop)
        
        dsSize = dataset.__len__().numpy()
        return self.__createDatasets(dataset, dsSize, batchSize)
    
    @staticmethod
    def preprocess_triplets(anchor: str, positive: str, negative: str) -> tuple[tf.Tensor,tf.Tensor,tf.Tensor]:
        return (
            DataLoader.decode_img(anchor),
            DataLoader.decode_img(positive),
            DataLoader.decode_img(negative),
        )
    
    @staticmethod
    def decode_img(img_path: str):
        image_size = (224, 224)
        num_channels = 3
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(
            img, channels=num_channels, expand_animations=False
        )
        img = tf.image.resize(img, image_size, method="bilinear")
        img.set_shape((image_size[0], image_size[1], num_channels))
        return img
    
    @staticmethod
    def decode_imgs(img_path1: str, img_path2: str, label: str):
        image_size = (224, 224)
        num_channels = 3
        img1 = tf.io.read_file(img_path1)
        img1 = tf.image.decode_image(
            img1, channels=num_channels, expand_animations=False
        )
        img1 = tf.image.central_crop(img1, 0.7)
        img1 = tf.image.resize(img1, image_size, method="bilinear")
        img1.set_shape((image_size[0], image_size[1], num_channels))
        
        img2 = tf.io.read_file(img_path2)
        img2 = tf.image.decode_image(
            img2, channels=num_channels, expand_animations=False
        )
        img2 = tf.image.central_crop(img2, 0.7)
        img2 = tf.image.resize(img2, image_size, method="bilinear")
        img2.set_shape((image_size[0], image_size[1], num_channels))
        return {"input_anchor": img1, "input_compare": img2}, label
    
    @staticmethod
    def decode_imgs_no_crop(img_path1: str, img_path2: str, label: str):
        image_size = (224, 224)
        num_channels = 3
        img1 = tf.io.read_file(img_path1)
        img1 = tf.image.decode_image(
            img1, channels=num_channels, expand_animations=False
        )
        img1 = tf.image.resize(img1, image_size, method="bilinear")
        img1.set_shape((image_size[0], image_size[1], num_channels))
        
        img2 = tf.io.read_file(img_path2)
        img2 = tf.image.decode_image(
            img2, channels=num_channels, expand_animations=False
        )
        img2 = tf.image.resize(img2, image_size, method="bilinear")
        img2.set_shape((image_size[0], image_size[1], num_channels))
        return {"input_anchor": img1, "input_compare": img2}, label
    
    @staticmethod
    def __readImage(image_file, label):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (224,224))
        return image, label
    
    def __unbatchDataset(self, ds: tf.data.Dataset) -> tuple[tf.data.Dataset, int]:
        dsSize = ds.__len__().numpy()
        ds = ds.unbatch()
        return (ds, dsSize)

    def __createDatasets(
        self, ds: tf.data.Dataset, dsSize: int, batchSize: int
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Split the dataset into train, validation und test. Create batches and init prefetching.

        Args:
            ds (tf.data.Dataset): dataset to split
            dsSize (int): size of the dataset
            batchSize (int): batch size for created datasets

        Returns:
            tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: datasets in the order train, validation, test
        """
        assert (self.trainSplit + self.testSplit + self.valSplit) == 1

        if self.shuffle:
            ds = ds.shuffle(2 * dsSize, seed=self.seed)

        trainSize = int(self.trainSplit * dsSize)
        valSize = int(self.valSplit * dsSize)

        train_ds = ds.take(trainSize)
        train_ds = train_ds.batch(batchSize)
        train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_ds = ds.skip(trainSize).take(valSize)
        val_ds = val_ds.batch(batchSize)
        val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        test_ds = ds.skip(trainSize).skip(valSize)
        test_ds = test_ds.batch(batchSize)
        test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

        return train_ds, val_ds, test_ds
