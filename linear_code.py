import numpy as np
import bitarray as ba
import pathlib


class HemmingCode:

    def create_map_err(self):
        H_trans = np.transpose(self.H)
        map_err = dict()
        for i in range(len(H_trans)):
            cur_column = str()
            for j in range(len(H_trans[0])):
                cur_column = cur_column + str(H_trans[i][j])
            else:  # after all success  iteration
                map_err[cur_column] = i

    def __init__(self, H: np.ndarray, G: np.ndarray):
        self.H = H  # проверочная матрица
        self.G = G  # порождающая матрица
        self.n = len(H)
        self.map_err = dict()
        self.n = len(H[0])  # endcoded count of bits
        self.k = self.n - len(H)  # information bits
        self.map_err = self.create_map_err()

    def endcode(self, bit_arr: ba.bitarray, is_reshaped: bool = False) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.k
        :param is_reshaped: flag is it array shape - (l,self.k), where l - some value greater 0
        :return: endcoded array multiple of self.n
        """
        bit_list = None
        if not is_reshaped:
            # create reshaped array
            bit_list = bit_arr.tolist()
            bit_list = np.asarray(bit_list, dtype=np.int8)
            bit_list = bit_list.reshape((len(bit_list) // self.k, self.k))
        else:
            bit_list = bit_arr

        coded_arr = ba.bitarray()
        # endcode
        for i in range(len(bit_list)):
            coded_arr.extend(np.matmul(bit_list[i], self.G) % 2)

        return coded_arr

    def decode(self, bit_arr: ba.bitarray, is_reshaped: bool = False) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.n
        :param is_reshaped: flag is it array shape - (l, self.n), where l - some value greater 0
        :return: decoded array multiple of self.k
        """
        bit_list = None
        if not is_reshaped:
            # create reshaped array
            bit_list = bit_arr.tolist()
            bit_list = np.asarray(bit_list, dtype=np.int8)
            bit_list = bit_list.reshape((len(bit_list) // self.n, self.n))
        else:
            bit_list = bit_arr

        bit_decoded = ba.bitarray()
        for i in range(len(bit_list)):
            bit_decoded.extend(bit_list[i][:self.k])

        return bit_decoded


def main():
    cur_path = pathlib.Path() / "lab1"

    G = [[1, 0, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 1, 0, 1, 1]]

    H = [[1, 1, 1, 0, 1, 0, 0],
         [1, 1, 0, 1, 0, 1, 0],
         [1, 0, 1, 1, 0, 0, 1]]


    G = np.asanyarray(G)
    H = np.asanyarray(H)
    cls_code = HemmingCode(H, G)
    # task 3
    print(f"n: {cls_code.n}\nk: {cls_code.k}")

    bit_arr = ba.bitarray()
    with open(cur_path / "inputfile", "rb") as file:
        bit_arr.fromfile(file)


    endcoded_ba = cls_code.endcode(bit_arr)
    with open("test_class.txt","wb")as file:
        endcoded_ba.tofile(file)


if __name__ == '__main__':
    main()
