# -*- coding: utf-8 -*-
import allspark
from models.model_config import ESIMConfig
from models.model import ESIM
from utils.load_data import char_index
import tensorflow as tf


class MyProcessor(allspark.BaseProcessor):
    """ MyProcessor is a example
        you can send mesage like this to predict
        curl -v http://127.0.0.1:8080/api/predict/service_name -d '2 105'
    """

    def initialize(self):
        """ load module, executed once at the start of the service
         do service intialization and load models in this function.
        """
        self.model_config = ESIMConfig()
        self.model = ESIM(self.model_config).get_model()
        self.model.load_weights('saved_models/esim_LCQMC_32_LSTM_0715_1036.h5')
        self.model._make_predict_function()
        global graph
        graph = tf.get_default_graph()

    def pre_proccess(self, data):
        """ data format pre process
        """
        x, y = data.split(b' ')
        #print(str(x), str(y))
        x, y = char_index([str(x, encoding="utf-8")], [str(y, encoding="utf-8")])
        # x, y = char_index([x], [y])
        return x, y

    def post_process(self, data):
        """ proccess after process
        """
        return str(data).encode()

    def process(self, data):
        """ process the request data
        """

        x, y = self.pre_proccess(data)
        global graph
        with graph.as_default():
            y_pred = self.model.predict([x, y]).item()

        # y_pred = 0

        return float(y_pred), 0


if __name__ == '__main__':
    # paramter worker_threads indicates concurrency of processing
    runner = MyProcessor(worker_threads=10)
    runner.run()
