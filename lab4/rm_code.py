import numpy as np
import bitarray as ba
import pathlib
import random

from static_meth_for_encode import BaseCode

# For correct working of the path system and so that the files that are used would be in this directory
cur_path = pathlib.Path()


class RM_Code(BaseCode):

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

        self.bone = dict()
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
    def get_H_i_m(i: int, m: int):
        # H^i_m = E_2^(m-i) x H x  E_2^(i-1)
        H = np.array([[1, 1],
                      [1, -1]], dtype=np.int8)
        return np.kron(np.kron(np.eye(2 ** (m - i), dtype=np.int8), H), np.eye(2 ** (i - 1), dtype=np.int8))

    def get_v_j(self, j) -> ba.bitarray:
        reverse_bit_num: str = BaseCode.int_to_str_bits(j, self.m)
        reverse_bit_num: ba.bitarray = ba.bitarray(reverse_bit_num)
        reverse_bit_num.reverse()
        return reverse_bit_num

    def decode(self, bit_arr: ba.bitarray) -> ba.bitarray:
        """
        :param bit_arr: bitarray that length multiple of self.n
        :param is_reshaped: flag is it array shape - into ndarray (l,self.k), where l - some value greater 0
        :return: decoded array multiple of self.k
        """
        # msg is being partited on pieces
        # print(bit_arr, len(bit_arr))
        bit_list = BaseCode.reshape_to_np_arr(bit_arr, self.n)
        decoded_arr = ba.bitarray()

        for i in range(len(bit_list)):

            w_edit = RM_Code.get_w_edit(bit_list[i])
            # w^~= w_edit = w_i
            for j in range(self.m):
                H_i_m = RM_Code.get_H_i_m(j + 1, self.m)
                w_edit = np.matmul(w_edit, H_i_m)

            max_index_j = np.argmax(np.abs(w_edit))
            vj: ba.bitarray = self.get_v_j(max_index_j)
            first_bit: tuple = (int(w_edit[max_index_j] > 0),)

            decoded_arr.extend(first_bit)
            decoded_arr.extend(vj)

        return decoded_arr

    def encode_file(self, file_in, file_out):
        """
        :param file_in:
        :param file_out:
        :return:
        """

        bit_a = ba.bitarray()
        bit_a.fromfile(file_in)
        len_bit_a = len(bit_a)

        print(f"our msg {len(bit_a)}, {bit_a}")
        bit_a = self.add_to_multiplicity_n(bit_a, self.k)
        print(f"our msg {len(bit_a)}, {bit_a}")

        if len(bit_a) - len_bit_a >= 8:
            self.bone[len(bit_a)] = len_bit_a

        encode = self.encode(bit_a)
        print(f"endcode {len(encode)}, {encode}")

        encode.tofile(file_out)

    def decode_file(self, file_in, file_out):
        """
        Have inner dependence from method encode_file
        Doesn't recomended to call it without calling the encode_file method.
        You can get back more bits what in really msg.

        Example
        You have n = 15, k=9
        and lenght of msg in bits is 136.
        To decode msg u need extend bitarry to 144 bit (144⋮ 9 144⋮ 8). And it also divide by 8 without remain.
        144 = 8 * 9 * 2
        And after decoding, u need delete byte at the end of msg

        And in other case u have msg with 144 bits lenght
        :param file_in:
        :param file_out:
        :return:
        """
        bit_a = ba.bitarray()
        bit_a.fromfile(file_in)

        print(f"endcode {len(bit_a)}, {bit_a}")
        bit_a = BaseCode.add_to_multiplicity_n(bit_a, self.n, is_to_less=True)
        print(f"endcode {len(bit_a)}, {bit_a}")

        decode = self.decode(bit_a)

        key = len(decode)
        true_len = self.bone.setdefault(key, None)
        if true_len:
            decode = decode[:true_len]
        # clear the memory
        del self.bone[key]

        print(f"decode  {len(decode)}, {decode}")
        # decode = self.add_to_multiplicity_n(decode, 8, is_to_less=True)
        decode.tofile(file_out)

    def make_err(self, bit_a: ba.bitarray):
        """
        :param bit_a: have length > 0
        :return:
        """
        rand_index = random.randint(0, len(bit_a) - 1)
        print(("Create err at index %.3d" % rand_index) + " " * rand_index + "|")

        bit_a[rand_index] = not bit_a[rand_index]
        return bit_a

    def make_err_file(self, file_in, file_out, make_err_function):
        bit_a = ba.bitarray()
        bit_a.fromfile(file_in)

        bit_a = make_err_function(bit_a)
        bit_a.tofile(file_out)


if __name__ == '__main__':
    r = 1
    m = 3
    main_obj = RM_Code(r, m)
    bit_a = ba.bitarray("1100")
    encoded = main_obj.encode(bit_a)
    # sec_obj = RM_Code(1,3)

    print("\nTask 4.2.2")
    print(f"\n G{r, m} = \n{main_obj.g}")
    print(f"k = {main_obj.k}, n = {main_obj.n}")
    # print(f"\n G{1, 3} = \n{sec_obj.g}")
    print(f"msg = {bit_a}")
    print(f"encoded msg  = {encoded}")

    print("\nTask 4.2.3")
    bit_a_with_err = ba.bitarray("10101011")
    decoded = main_obj.decode(bit_a_with_err)

    print(f"msg with err = {bit_a_with_err}")
    print(f"decoded msg = {decoded}")

    print("\nTask 4.2.4")
    with open(cur_path / "inputfile", "rb") as file_in:
        with open(cur_path / "output", "wb") as file_out:
            main_obj.encode_file(file_in, file_out)

        with open(cur_path / "output", "rb") as file_out:
            with open(cur_path / "err", "wb") as file_err:
                main_obj.make_err_file(file_out, file_err, main_obj.make_err)

        with open(cur_path / "output7", "wb") as file_out:
            with open(cur_path / "err", "rb") as file_err:
                main_obj.decode_file(file_err, file_out)

    print("")
