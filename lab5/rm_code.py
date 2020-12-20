import numpy as np
import bitarray as ba
import pathlib
import random
from itertools import combinations

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

    # 5.2.1
    def get_J(self):
        array = [()]  # list with empty tuple
        # get int val from reversed tuple elem casted to str
        key = lambda arr: int(''.join(reversed([str(i) for i in arr])))
        for i in range(0, self.r):
            non_sorted_list = list(combinations(range(self.m), i + 1))  # combination from range of len = i+1
            sorted_list = sorted(non_sorted_list, reverse=True, key=key)
            array.extend(sorted_list)
        return array

    # 5.2.2
    def get_bit_view_nums(self):
        ans = []
        for i in range(self.n):
            str_bits = BaseCode.int_to_str_bits(i, self.m)
            bit_a = ba.bitarray(str_bits)
            bit_a.reverse()
            ans.append(bit_a)
        return ans

    # 5.2.3
    def get_matrix(self):
        matrix = np.ndarray((len(self.J), self.n), dtype=np.int8)
        for i, j_el in enumerate(self.J):
            matrix[i][:] = self.get_row_by_j(j_el)[:]
        return matrix

    def __init__(self, r: int, m: int):
        if r > m:
            raise ValueError(f"r must be less or equal m. r ={r}, m = {m}")

        self.r: int = r
        self.m: int = m

        self.bone = dict()
        self.k: int = self.get_k()
        self.n: int = 2 ** self.m
        self.J = self.get_J()
        self.bin_num = self.get_bit_view_nums()
        self.matrix = self.get_matrix()

    # 5.2.3 | 5.3.1.2
    def get_row_by_j(self, j):
        row = np.ones(self.n, dtype=np.int8)
        for i, bit in enumerate(self.bin_num):
            if any([bit[el] != 0 for el in j]):
                row[i] = 0
        return row

    # 5.3.1.1
    def get_complementary_set(self, j: tuple):
        return set(range(self.m)) - set(j)

    # 5.3.1.3
    def get_permissible_bit_shift(self, J):
        views = [list(i) for i in self.bin_num.copy()]
        for t in views:
            for j in J:
                t[j] = 0
        views = np.unique(views, axis=0)
        return views

    # 5.3.1.4
    def get_verification_vectors(self, J):
        J_c = self.get_complementary_set(J)
        shifts = self.get_permissible_bit_shift(J)
        shifts = [BaseCode.get_num_from_bit(shift[::-1]) for shift in shifts]
        row = self.get_row_by_j(J_c)
        ans = np.ndarray((len(shifts), self.n), dtype=np.int8)
        for i, shift in enumerate(shifts):
            shifted_row = row.copy()
            if shift != 0:
                shifted_row[shift:] = shifted_row[:-shift]
                shifted_row[:shift] = np.zeros(shift, dtype=np.int8)
            ans[i][:] = shifted_row
        return ans

    @staticmethod
    def scalar(a, b):
        multiply = a * b
        sum = 0
        for i in multiply:
            sum ^= i
        return sum

    # 5.3.2
    def decode(self, bit_arr: ba.bitarray):

        bit_list = BaseCode.reshape_to_np_arr(bit_arr, self.n)
        decoded_arr = ba.bitarray()

        for kij in range(len(bit_list)):
            i = self.r
            J = self.J
            w = bit_list[kij].copy()
            decoded = np.zeros(self.k, dtype=np.int8)
            v_i = np.zeros(self.n, dtype=np.int8)
            for j in range(len(J) - 1, -1, -1):
                J_j = J[j]
                if len(J_j) != i:
                    i -= 1
                    w = np.logical_xor(w, v_i)
                    v_i = np.zeros(self.n, dtype=np.int8)
                count = 0
                verifications = self.get_verification_vectors(J_j)
                for v in verifications:
                    count += RM_Code.scalar(w, v)
                size_half = len(verifications) // 2
                if count > size_half:
                    decoded[j] = 1
                    v_i = np.logical_xor(v_i, self.get_row_by_j(J_j))
                elif count == size_half:
                    print(f"Error. The word {w} can be fixed.")
            decoded_arr.extend(decoded)

        return decoded_arr

    # 5.2.4 |
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
        log_xor = np.matmul(bit_list, self.matrix) % 2
        coded_arr.extend(log_xor.ravel())
        return coded_arr


def main():
    r = 2
    m = 4
    main_obj = RM_Code(2, 4)
    print("\n\nTask 5.2.0")
    print(f"r = {r},m = {m},k ={main_obj.k}")

    print("\n\nTask 5.2.1")
    print("J ", np.array(main_obj.J))

    print("\n\nTask 5.2.2")
    print("bit ", main_obj.bin_num)

    print("\n\nTask 5.2.3")
    print(f"G{r, m} \n", np.array(main_obj.matrix))

    msg = "1 0 0 0 0 0 0 1 0 0 0".replace(" ", "")
    bit_a = ba.bitarray(msg)
    encoded = main_obj.encode(bit_a)

    print("\n\nTask 5.2.4")
    print(f"msg = {bit_a}\nendcoded = {encoded}")

    print("\n\nTask 5.3.1.3")
    J = (1, 3)
    print(f"for J= {J} {main_obj.get_permissible_bit_shift(J)}")

    print("\n\nTask 5.3.1.4")
    print(f"for J= {J} \n{main_obj.get_verification_vectors(J)}")

    word = "1 0 1 0 1 0 0 0 0 1 0 1 1 1 1 1".replace(" ", "")
    bit_a_word = ba.bitarray(word)
    decoded = main_obj.decode(bit_a_word)

    print("\n\nTask 5.3.2")
    print(f"encoded word = {bit_a_word}")
    print(f"decoded word = {decoded}")


if __name__ == '__main__':
    main()
