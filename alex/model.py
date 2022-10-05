import numpy as np
import transformers
import tensorflow as tf


class FeedbackPrizeModel():
    def __init__(self):
        # BERT type encoder:
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        self.max_len = 512

    def encoder(self, texts):
        input_ids = []
        token_type_ids = []
        attention_mask = []

        for text in texts:
            tokenizer = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length',
                                       add_special_tokens=True)
            tokenizer.save_pretrained('.')
            input_ids.append(tokenizer['input_ids'])
            token_type_ids.append(tokenizer['token_type_ids'])
            attention_mask.append(tokenizer['attention_mask'])

        return np.array(input_ids), np.array(token_type_ids), np.array(attention_mask)

    def get_model(self):
        # Input layers:
        input_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="input_ids")
        token_type_ids = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="token_type_ids")
        attention_mask = tf.keras.layers.Input(shape=(self.max_len,), dtype=tf.int32, name="attention_mask")

        # Inner layers:
        transformer = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        clf_output = transformer[:, 0, :]

        # Output layer:
        output = tf.keras.layers.Dense(5, activation='softmax')(clf_output)

        # Init model and compile model:
        model = tf.keras.Model(inputs=[input_ids, token_type_ids, attention_mask], outputs=output)
        model.compile(tf.optimizers.Adam(lr=3e-5), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy', 'sparse_categorical_crossentropy'])

        return model

