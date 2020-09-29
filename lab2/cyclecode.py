import numpy as np
import bitarray as ba
import pathlib


class CyclicCode:
    def __init__(self, g: ba.bitarray, n: int = 7, k: int = 4):
        self.g = g
        self.list_g = g.tolist()
        self.np_g = np.asarray(self.list_g, dtype=np.int8)
        self.k = k
        self.n = n
        self.n_k = n - k  # n minus k
        self.extended_g = self.list_g.copy()
        self.extended_g.extend([0] * (n - k))

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
                bit_arr = np.asarray(bit_arr, dtype=np.int8)
            bit_arr = bit_arr.reshape((len(bit_arr) // k, k))

        return bit_arr

    def endcode(self, bit_arr: (ba.bitarray, np.array, np.ndarray), is_reshaped: bool = False) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.k
        :param is_reshaped: flag is it array shape - (l,self.k), where l - some value greater 0
        :return: endcoded array multiple of self.n
        """
        bit_list = self.reshape_to_np_arr(bit_arr, self.k, is_reshaped)
        coded_arr = ba.bitarray()
        # endcode
        for i in range(len(bit_list)):
            curr_np = np.asarray([False] * self.n)
            for j in range(self.k):
                if bit_list[i][j]:
                    xj_g = [False] * j
                    xj_g.extend(self.list_g)
                    xj_g.extend([False] * (self.n_k - j))
                    xj_g = np.asarray(xj_g)
                    curr_np = np.logical_xor(curr_np, xj_g)
            else:
                coded_arr.extend(curr_np)

        return coded_arr

    def remain_dev(self, bit_arr: (ba.bitarray, np.array, np.ndarray), is_reshaped: bool = False):
        """
                :param bit_arr: bitarray that length multiple of self.k
                :param is_reshaped: flag is it array shape - (l,self.k), where l - some value greater 0
                :return: endcoded array multiple of self.n
                """
        bit_list = self.reshape_to_np_arr(bit_arr, self.n, is_reshaped)
        coded_arr = ba.bitarray()
        # print(f"bit_list: {bit_list}")
        # endcode
        for i in range(len(bit_list)):
            curr_np = bit_list[i]
            # print(f"bit_list[{i}] { bit_list[i]} ")
            for j in range(self.n - 1, self.n_k - 1, -1):
                # print(f"for j ={j}\n")
                if curr_np[j]:
                    # len(list_g) === k
                    # n=7. j=6. len(g) = k = 4.
                    xj_g = [False] * (j - (self.k - 1))
                    xj_g.extend(self.list_g)
                    xj_g.extend([False] * ((self.n - 1) - j))
                    xj_g = np.asarray(xj_g)
                    curr_np = np.logical_xor(curr_np, xj_g)
                    # print(f"curr_np: {curr_np} \n")
            else:
                coded_arr.extend(curr_np)

        return coded_arr


def main():
    g = [1, 0, 1, 1]
    bit_g = ba.bitarray()
    bit_g.extend(g)

    a = [1, 0, 1, 0]
    bit_a = ba.bitarray()
    bit_a.extend(a)

    main_obj = CyclicCode(bit_g)

    # task 1
    print("\n\nTask1")
    encode_a = main_obj.endcode(bit_a)
    print(f"endcode a: {encode_a}")

    # task 2
    print("\n\nTask2")
    a2 = [1, 0, 1, 0, 1, 0, 1]
    bit_a2 = ba.bitarray()
    bit_a2.extend(a2)

    remain_e_a = main_obj.remain_dev(encode_a)
    remain_a2 = main_obj.remain_dev(bit_a2)

    print(f"remain_dev a: {remain_e_a}")
    print(f"bit_a2: {a2}\nremain_dev a2: {remain_a2}")

    # task 2*
    print("\n\nTask2*")
    identity = np.identity(7, dtype=np.int8).ravel()
    remain_dev = main_obj.remain_dev(identity)
    remain_dev = main_obj.reshape_to_np_arr(remain_dev, 7)
    print(f"remain_dev: \n{remain_dev}")


if __name__ == '__main__':
    main()
