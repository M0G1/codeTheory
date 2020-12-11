import numpy as np
import bitarray as ba
import pathlib
import random

from static_meth_for_encode import BaseCode


class RM_Code:

    def get_k(self) -> int:
        k_sum = 0.0
        m_fac = np.math.factorial(self.m)
        for i in range(self.r + 1):
            k_sum += m_fac / (np.math.factorial(self.m - i) * np.math.factorial(i))
        return int(k_sum)

    @staticmethod
    def get_g(r, m):
        if r == 0:
            # G(0,1) = [11...1]
            return np.ones((1, 2 ** m))
        if r == m:
            #          |G(m-1,m)|
            # G(m,m) = | 0...01|
            down = np.zeros((1, 2 ** m))
            down[0][-1] = 1
            return np.concatenate([RM_Code.get_g(m - 1, m), down])
        #           |G(r,m-1)  G(r,m-1)|
        #   G(r,m)= |0      G(r-1,m-1)|
        g_up = np.concatenate([RM_Code.get_g(r, m - 1), RM_Code.get_g(r, m - 1)], axis=1)
        g_down = RM_Code.get_g(r - 1, m - 1)
        return np.concatenate([g_up, np.concatenate([np.zeros(np.shape(g_down)), g_down], axis=1)])

    def __init__(self, r: int, m: int):
        if r > m:
            raise ValueError(f"r must be less or equal m. r ={r}, m = {m}")

        self.r: int = r
        self.m: int = m

        self.k: int = self.get_k()
        self.n: int = 2 ** self.m
        self.g = RM_Code.get_g(r, m)

    def encode(self, bit_arr: ba.bitarray, is_reshaped=False) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.k
        :param is_reshaped: flag is it array shape - into ndarray (l,self.k), where l - some value greater 0
        :return: encoded array multiple of self.n
        """
        # msg is being partited on pieces
        bit_list = BaseCode.reshape_to_np_arr(bit_arr, self.k, is_reshaped)
        coded_arr = ba.bitarray()
        # encode
        log_xor = np.matmul(bit_list, self.g) % 2
        coded_arr.extend(log_xor.ravel())
        return coded_arr

    @staticmethod
    def get_w_edit(w: (np.array, list, np.ndarray)):
        cond = w > 0
        return np.where(cond, 1, -1)

    @staticmethod
    def get_H_i_m(i, m):
        # H^i_m = E_2^(m-i) x H x  E_2^(i-1)
        H = np.array([[1, 1],
                      [1, -1]])
        return np.kron(np.kron(np.eye(2 ** (m - i)), H), np.eye(2 ** (i - 1)))

    def get_v_j(self, j):
        main_part = bin(j)[2:]

        print()

    def decode(self, bit_arr: ba.bitarray) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.n
        :param is_reshaped: flag is it array shape - into ndarray (l,self.k), where l - some value greater 0
        :return: decoded array multiple of self.k
        """
        # msg is being partited on pieces
        print(bit_arr, len(bit_arr))
        bit_list = BaseCode.reshape_to_np_arr(bit_arr, self.n)
        decoded_arr = ba.bitarray()

        for i in range(len(bit_list)):
            w_edit = RM_Code.get_w_edit(bit_list[i])
            # w^~= w_edit = w_i
            for j in range(self.m):
                w_edit = np.matmul(w_edit, RM_Code.get_H_i_m(i + 1, self.m))

            max_index = np.argmax(np.abs(w_edit))

        return decoded_arr


if __name__ == '__main__':
    r = 1
    m = 3
    main_obj = RM_Code(r, m)
    bit_a = ba.bitarray("1100")
    encoded = main_obj.encode(bit_a)
    # sec_obj = RM_Code(1,3)

    print("Task 4.2.1")
    print(f"\n G{r, m} = \n{main_obj.g}")
    print(f"k = {main_obj.k}, n = {main_obj.n}")
    # print(f"\n G{1, 3} = \n{sec_obj.g}")
    print(f"msg = {bit_a}")
    print(f"encoded msg = {encoded}")

    print("")
