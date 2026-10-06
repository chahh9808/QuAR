# Added to the vendored AdaLog (github.com/GoatWu/AdaLog, commit 75a57d2), Apache-2.0: 8-bit configuration written for this project.

class Config:
    def __init__(self):
        # calibration settings
        self.calib_size = 32
        self.optim_size = 1024
        self.calib_batch_size = 32
        self.optim_batch_size = 32
        self.w_bit = 8
        self.a_bit = 8
        self.s_bit = 8
        self.qconv_a_bit = 8
        self.qhead_a_bit = 8
        self.matmul_head_channel_wise = True
        self.post_softmax_quantizer = 'adalog'
        self.post_gelu_quantizer = 'adalog'
        # search settings
        self.eq_n = 128
        self.search_round = 3
        self.fpcs = True
        self.steps = 6
        # optimization settings
        self.keep_gpu = True
        self.train_act = True
