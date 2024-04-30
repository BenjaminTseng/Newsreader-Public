import modal

# preprocessor preload baked into image construction
def download_preprocessor():
    import keras_nlp 
    base_model_id = "roberta_base_en"
    preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(base_model_id)

tf_image = (
    modal.Image.from_registry('tensorflow/tensorflow:2.16.1')
    .pip_install('keras==3.3.2')
    .pip_install('keras-nlp==0.9.3')
    .run_function(download_preprocessor)
)
app = modal.App("newsreader-recommend", image=tf_image)
vol = modal.Volume.from_name("newsreader-data")

# use Modal class to reduce cold start times by enabling memory snapshot and using @modal.enter to preload weights
@app.cls(volumes={"/data": vol}, enable_memory_snapshot=True, cpu=8.0, timeout=1000, retries=2)
class NewsreaderRecommendation:
    @modal.enter(snap=True)
    def enter(self):
        vol.reload() # make sure to load most recent volume
        # eliminate always there Tensorflow warnings from logs
        import os 
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        import warnings 
        warnings.filterwarnings('ignore')
        
        import keras 
        import keras_nlp

        self.base_model_id = "roberta_base_en"
        self.preprocessor = keras_nlp.models.RobertaPreprocessor.from_preset(self.base_model_id)

        with open('/data/model/best_model.txt', 'r') as f:
            model_path = '/data/model/'+f.read().strip()
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.quantize('int8') # quantize model to reduce memory impact / boost performance
        
        # save model surgery results for actual usage at run-time
        self.rating_model = keras.Model(inputs=self.model.inputs[2:], outputs=[self.model.outputs[1], self.model.get_layer('neural_collaborative_filter').input[0]], name='token_recommendation_model')
        self.vector_model = keras.Model(inputs=self.model.get_layer('neural_collaborative_filter').input, outputs=self.model.outputs[1], name='vector_recommendation_model')
        self.user_embeddings = self.model.get_layer('user_preferences').get_weights()[0]
        self.text_embedding = self.model.get_layer('backbone_model')

    # generates article embeddings from list of articles    
    @modal.method()
    def embedText(self, text_array):
        import keras
        text_tensors = keras.ops.convert_to_tensor(text_array)
        preprocessed = self.preprocessor(text_tensors)
        return self.text_embedding.predict([preprocessed['token_ids'], preprocessed['padding_mask']])

    # returns user embeddings
    @modal.method()
    def extractUserEmbeddings(self):
        return self.user_embeddings
    
    # rates list of text for list of users
    # returns article embeddings and list of ratings for each article (for each user)
    @modal.method()
    def rateTextUsers(self, text_array, user_id_list):
        import keras
        text_tensors = keras.ops.convert_to_tensor(text_array)
        preprocessed = self.preprocessor(text_tensors)
        ratings = []
        for user_id in user_id_list:
            user_rating_set, text_embeddings = self.rating_model.predict([preprocessed['token_ids'], preprocessed['padding_mask'], keras.ops.ones(text_tensors.shape[0])*user_id])
            ratings.append(user_rating_set)
        return ratings, text_embeddings 

    # performs rateTextUsers task but on article embeddings and user embeddings directly
    @modal.method()
    def rateVectors(self, text_vector_array, user_vector_list):
        import keras
        ratings = []
        text_tensors = keras.ops.convert_to_tensor(text_vector_array)
        for user_vector in user_vector_list:
            user_tensor = keras.ops.stack([user_vector for i in range(text_tensors.shape[0])])
            ratings.append(self.vector_model.predict([text_tensors, user_tensor]))
        return ratings 
