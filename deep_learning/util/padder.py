import numpy as np

class Padder:

    @classmethod
    def get_padding(cls, input_size, kernel_size, stride, padding):
        if padding == "valid":
            return cls.valid_padding()
        elif padding == "same":
            return cls.same_padding(
                    np.array(input_size[:-1]),
                    np.array(kernel_size),
                    np.array(stride))
        else:
            return cls.custom_padding(padding)


    @classmethod
    def valid_padding(cls):
        return  4*((0, 0),)


    @classmethod
    def same_padding(cls, input_size, kernel_size, stride):
        total_padding = input_size*(stride - 1) + kernel_size - stride
        left_padding  = total_padding//2
        right_padding = total_padding - left_padding

        return tuple(zip(left_padding, right_padding)) + 2*((0, 0),)


    @classmethod
    def custom_padding(cls, padding):
        return tuple((pad, pad) for pad in padding) + 2*((0, 0),)


