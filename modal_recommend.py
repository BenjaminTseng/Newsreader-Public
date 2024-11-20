import modal

jax_image = (
    modal.Image.debian_slim(python_version='3.11')
    .pip_install('jax[cuda12]==0.4.35', extra_options="-U")
    .pip_install('keras==3.6')
    .pip_install('keras-hub==0.17')
    .env({"KERAS_BACKEND":"jax"}) # sets environmental variable so switches to JAX
    .env({"XLA_PYTHON_CLIENT_MEM_FRACTION":"1.0"})
)
app = modal.App("newsreader-recommend", image=jax_image)
vol = modal.Volume.from_name("newsreader-data")

# use Modal class to reduce cold start times by enabling memory snapshot and using @modal.enter to preload weights
@app.cls(volumes={"/data": vol}, enable_memory_snapshot=True, timeout=1800, container_idle_timeout=120, retries=2)
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
        import keras_hub
        import json

        with open('/data/model/train_params.json', 'r') as f:
            trainparams = json.load(f)
            model_id = trainparams['model_id']

        self.preprocessor = keras_hub.models.TextClassifierPreprocessor.from_preset(model_id)

        with open('/data/model/best_model.txt', 'r') as f:
            model_path = '/data/model/'+f.read().strip()
        self.model = keras.models.load_model(model_path, compile=False)
        self.model.quantize('int8') # quantize model to reduce memory impact / boost performance
        
        # save model surgery results for actual usage at run-time
        self.user_embedding = keras.Model(inputs=self.model.inputs[2], outputs=self.model.get_layer('rating_out').input[1])
        self.text_embedding = self.model.get_layer('backbone_model')

    # generates article embeddings from list of articles    
    @modal.method()
    def embedText(self, text_array):
        import numpy as np 
        
        token_lengths = np.array([len(tokens) for tokens in self.preprocessor.tokenizer(text_array)])
        preprocessed = self.preprocessor(text_array)
        return self.text_embedding.predict([preprocessed['token_ids'], preprocessed['padding_mask']]), token_lengths

    # returns user embeddings
    @modal.method()
    def embedUser(self, user_ids):
        import keras
        return self.user_embedding.predict(keras.ops.convert_to_tensor(user_ids))
    
    # rates list of text for list of users
    # returns article embeddings and list of ratings for each article (for each user)
    @modal.method()
    def rateTextUsers(self, text_array, user_ids):
        import numpy as np
        import keras

        # sub in embedText code
        token_lengths = np.array([len(tokens) for tokens in self.preprocessor.tokenizer(text_array)])
        preprocessed = self.preprocessor(text_array)
        text_vec = self.text_embedding.predict([preprocessed['token_ids'], preprocessed['padding_mask']])

        # sub in embedUser
        user_vec = self.user_embedding.predict(keras.ops.convert_to_tensor(user_ids))
        
        ratings = 0.5 + 0.5*np.matmul(
            text_vec/np.linalg.norm(text_vec, axis=-1, keepdims=True),
            user_vec.T/np.linalg.norm(user_vec, axis=-1,keepdims=True).T
        ).T
        return ratings, text_vec, token_lengths 