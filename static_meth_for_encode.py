import numpy as np
import bitarray as ba
import pathlib
import random


class BaseCode:
    """
    Include base static method for working with encode/code.
    Use the list, numpy arrays and bitarray.
    """

    @staticmethod
    def reshape_to_np_arr(bit_arr: (ba.bitarray, np.array, np.ndarray), k: int, is_reshaped: bool = False,
                          dtype=np.int8):
        """
        :param bit_arr: bitarray that length multiple of k
        :param k: integer greater 0
        :return: reshaped np array
        """
        if not is_reshaped:
            # create reshaped array
            if isinstance(bit_arr, ba.bitarray):
                bit_arr = bit_arr.tolist()
            if isinstance(bit_arr, list):
                bit_arr = np.asarray(bit_arr, dtype=dtype)
            bit_arr = bit_arr.reshape((len(bit_arr) // k, k))

        return bit_arr

    @staticmethod
    def ba_to_str_bits(bit_arr: ba.bitarray) -> str:
        return str(bit_arr)[10:-2]

    @staticmethod
    def get_num_from_bit(bits: (str, list)):
        """
        bit order. least significant bit is right
        "110" -> 6
        :param bits:
        :return:
        """
        if isinstance(bits,np.ndarray):
            bits = list(bits)
        if isinstance(bits, list):
            bits = BaseCode.ba_to_str_bits(ba.bitarray(bits))
        return int(bits, 2)

    @staticmethod
    def int_to_str_bits(num: int, n) -> str:
        # n - needed length of return str
        if n == 0:
            return ""
        main_part = bin(num)[2:]
        return "0" * (n - len(main_part)) + main_part

    @staticmethod
    def add_to_multiplicity_n(bit_a: ba.bitarray, n: int, is_to_less: bool = False):
        rem_div_len = len(bit_a) % n
        if rem_div_len != 0:
            if is_to_less:
                bit_a = bit_a[:len(bit_a) - rem_div_len]
            else:
                bit_a = bit_a + ("0" * (n - rem_div_len))
        return bit_a

    @staticmethod
    def wt(word: (list, np.array, np.ndarray, ba.bitarray), is_np_array=False):
        # Hamming weight
        np_array = BaseCode.reshape_to_np_arr(word, len(word), is_reshaped=is_np_array)
        return np.count_nonzero(np_array)
