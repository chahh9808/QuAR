# Added to the vendored AdaLog (github.com/GoatWu/AdaLog, commit 75a57d2), Apache-2.0: W4A8 configuration written for this project.

class Config:
    """W4A8: 4-bit weights with 8-bit activations.

    Copied from configs/4bit.py with only the activation-side bit-widths raised
    to 8 (a_bit, s_bit, qhead_a_bit). Weights, the search settings and the
    optimization settings are unchanged, so this differs from the shipped 4-bit
    configuration in exactly the activation precision.
    """
    def __init__(self):
        # calibration settings
        self.calib_size = 32
        self.optim_size = 1024
        self.calib_batch_size = 32
        self.optim_batch_size = 32
        self.w_bit = 4
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
