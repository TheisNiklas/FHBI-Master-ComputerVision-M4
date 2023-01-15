import tensorflow as tf
from typing import Any

from utils.DistanceLayer import DistanceLayer


class ModelLoader:
    imgHeight: int
    imgWidth: int
    imgDepth: int

    def __init__(self, imgHeight: int = 224, imgWidth: int = 224, imgDepth: int = 3):
        self.imgHeight = imgHeight
        self.imgWidth = imgWidth
        self.imgDepth = imgDepth

    def loadMobileNetV1FaceRecognition(self, freezeBaseModelComplete: bool = True) -> tuple[tf.keras.Model, tf.keras.Model]:
        
        baseModel: tf.keras.Model = tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights="imagenet",  # type: ignore
        )
        if freezeBaseModelComplete:
            baseModel.trainable = False
        else:
            trainable = False
            for layer in baseModel.layers:
                if layer.name == "conv_dw_12":
                    trainable = True
                layer.trainable = trainable
        
        inputs = tf.keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgDepth), name="input")
        preprocess_layer = tf.keras.applications.mobilenet.preprocess_input(inputs)
        model: tf.keras.Model = baseModel(preprocess_layer)
        
        feature_extractor = tf.keras.layers.Flatten()(baseModel.output)
        feature_extractor = tf.keras.layers.Dense(1, activation="sigmoid")(feature_extractor)
        feature_extractor = tf.keras.layers.Dense(512, activation="relu")(feature_extractor)
        feature_extractor = tf.keras.layers.BatchNormalization()(feature_extractor)
        feature_extractor = tf.keras.layers.Dense(256, activation="relu")(feature_extractor)
        feature_extractor = tf.keras.layers.BatchNormalization()(feature_extractor)
        output = tf.keras.layers.Dense(256)(feature_extractor)
        
        embedding = tf.keras.Model(baseModel.input, output, name="Embedding")
        
        anchor_input = tf.keras.layers.Input(name="anchor", shape=(self.imgHeight, self.imgWidth, self.imgDepth))
        positive_input = tf.keras.layers.Input(name="positive", shape=(self.imgHeight, self.imgWidth, self.imgDepth))
        negative_input = tf.keras.layers.Input(name="negative", shape=(self.imgHeight, self.imgWidth, self.imgDepth))

        distances = DistanceLayer()(
            embedding(tf.keras.applications.mobilenet.preprocess_input(anchor_input)),
            embedding(tf.keras.applications.mobilenet.preprocess_input(positive_input)),
            embedding(tf.keras.applications.mobilenet.preprocess_input(negative_input)),
        )

        siamese_network = tf.keras.Model(
            inputs=[anchor_input, positive_input, negative_input], outputs=distances
        )
        
        return siamese_network, embedding
    
    def loadMobileNetV1FaceRecognitionPair(self) -> tuple[tf.keras.Model, tf.keras.Model]:
        
        baseModel: tf.keras.Model = tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights="imagenet",  # type: ignore
        )
        
        baseModel.trainable = False
        
        
        inputs = tf.keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgDepth), name="input")
        preprocess_layer = tf.keras.applications.mobilenet.preprocess_input(inputs)
        model: tf.keras.Model = baseModel(preprocess_layer)

        
        feature_extractor = tf.keras.layers.Flatten()(baseModel.output)
        feature_extractor = tf.keras.layers.Dense(512, activation="relu")(feature_extractor)
        feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)
        feature_extractor = tf.keras.layers.BatchNormalization()(feature_extractor)
        feature_extractor = tf.keras.layers.Dense(256, activation="relu")(feature_extractor)
        feature_extractor = tf.keras.layers.Dropout(0.2)(feature_extractor)
        feature_extractor = tf.keras.layers.BatchNormalization()(feature_extractor)
        output = tf.keras.layers.Dense(256)(feature_extractor)
        
        embedding = tf.keras.Model(baseModel.input, output, name="Embedding")
        
        img1 = tf.keras.layers.Input(shape=(self.imgWidth, self.imgHeight, self.imgDepth))
        img2 =  tf.keras.layers.Input( shape=(self.imgWidth, self.imgHeight, self.imgDepth))
        featsA = embedding(img1)
        featsB = embedding(img2)
        
        distance = tf.keras.layers.Lambda(ModelLoader.euclidean_distance)([featsA, featsB])
        
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
        model = tf.keras.Model(inputs=[img1, img2], outputs=outputs)
        
        return model, embedding
    
    @staticmethod
    def euclidean_distance(vectors):
        (featsA, featsB) = vectors
        sumSquared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1,keepdims=True)
        return tf.keras.backend.sqrt(tf.keras.backend.maximum(sumSquared, tf.keras.backend.epsilon()))
    
    def loadMobileNetV1(
        self,
        train_ds: tf.data.Dataset,
        freezeBaseModel: bool = False,
        initWeightsRandom: bool = True,
        nodeCount: int = 1
    ) -> tf.keras.Model:
        """
        Loads MobileNetV1 with custom top

        Args:
            train_ds (tf.data.Dataset): training dataset
            freezeBaseModel (bool, optional): whether to freeze the weights of the base model. Defaults to False.
            initWeightsRandom (bool, optional): initialize weights random if True or load imagenet weights if False. Defaults to True.

        Returns:
            tf.keras.Model
        """
        weights = None
        if not initWeightsRandom:
            weights = "imagenet"
        baseModel: tf.keras.Model = tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights=weights,  # type: ignore
        )
        if freezeBaseModel:
            baseModel.trainable = False

        model = self.__buildModel(
            baseModel, train_ds, tf.keras.applications.mobilenet.preprocess_input, nodeCount
        )

        return model
    
    def loadMobileNetV1Age(
        self,
        train_ds: tf.data.Dataset,
        freezeBaseModel: bool = False,
        initWeightsRandom: bool = True,
        nodeCount: int = 1
    ) -> tf.keras.Model:
        """
        Loads MobileNetV1 with custom top

        Args:
            train_ds (tf.data.Dataset): training dataset
            freezeBaseModel (bool, optional): whether to freeze the weights of the base model. Defaults to False.
            initWeightsRandom (bool, optional): initialize weights random if True or load imagenet weights if False. Defaults to True.

        Returns:
            tf.keras.Model
        """
        weights = None
        if not initWeightsRandom:
            weights = "imagenet"
        baseModel: tf.keras.Model = tf.keras.applications.MobileNet(
            input_shape=(self.imgHeight, self.imgWidth, self.imgDepth),
            include_top=False,
            weights=weights,  # type: ignore
        )
        if freezeBaseModel:
            baseModel.trainable = False

        model = self.__buildModel(
            baseModel, train_ds, tf.keras.applications.mobilenet.preprocess_input, nodeCount, "softmax"
        )

        return model

    def loadTrainedModel(self, modelPath: str) -> tf.keras.Model:
        return tf.keras.models.load_model(modelPath)

    def __getGlobalAverageLayer(self) -> tf.keras.layers.GlobalAveragePooling2D:
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        return global_average_layer

    def __getPredictionLayer(self, nodeCount: int, activation: str) -> tf.keras.layers.Dense:
        prediction_layer = tf.keras.layers.Dense(nodeCount, activation=activation)
        return prediction_layer

    def __buildModel(
        self, baseModel: tf.keras.Model, train_ds: tf.data.Dataset, preprocess, nodeCount: int, activation: str = "softmax"
    ) -> tf.keras.Model:
        global_average_layer = self.__getGlobalAverageLayer()

        inputs = tf.keras.Input(shape=(self.imgWidth, self.imgHeight, self.imgDepth))
        x = preprocess(inputs)
        x = baseModel(x)
        x = global_average_layer(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = self.__getPredictionLayer(nodeCount, activation)(x)
        model = tf.keras.Model(inputs, outputs)
        return model
