import numpy as np
import bitarray as ba
import pathlib
import random

# For correct working of the path system and so that the files that are used would be in this directory
cur_path = pathlib.Path()


class CyclicCode:
    def __init__(self, g: ba.bitarray, n: int = 7, k: int = 4, t: int = 1, L: int = 0):
        if n - k + 1 != len(g):
            raise AttributeError(f"Polynomial g must have lenght n - k: {n} - {k} = {n - k}")
        self.g = g
        self.list_g = g.tolist()
        self.np_g = np.asarray(self.list_g, dtype=np.int8)
        self.k = k
        self.n = n
        self.t = t
        self.L = L
        self.bone = dict()
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
                bit_arr = np.asarray(bit_arr, dtype=dtype)
            bit_arr = bit_arr.reshape((len(bit_arr) // k, k))

        return bit_arr

    def encode(self, bit_arr: ba.bitarray, is_reshaped: bool = False) -> ba.bitarray:
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

    def remain_dev(self, bit_arr: ba.bitarray, is_reshaped: bool = False) -> ba.bitarray:
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
                    curr_np[j - self.n_k:j + 1] = np.logical_xor(curr_np[j - self.n_k:j + 1], self.np_g)
            else:
                coded_arr.extend(curr_np)

        return coded_arr

    def encody_sys(self, bit_arr: ba.bitarray, is_reshaped: bool = False):
        """
        :param bit_arr:
        :param is_reshaped:
        :return:
        """
        # print(len(bit_arr))
        encody = ba.bitarray()

        for i in range(len(bit_arr) // self.k):
            c = ba.bitarray(self.n)
            c.setall(False)
            # a = bit_arr[i * self.k:(i + 1) * self.k]
            c[self.n_k:self.n] = bit_arr[i * self.k:(i + 1) * self.k]
            r = self.remain_dev(c)
            c[0:self.n_k] = r[0:self.n - self.k]
            encody.extend(c)

        return encody

    def decode_sys(self, bit_arr: ba.bitarray, make_table, is_fix_err: bool = True):
        """
        With correct.
        :param bit_arr:
        :param is_reshaped:
        :return:
        """
        bit_arr = self.add_to_multiplicity_n(bit_arr, self.n)
        decoded = ba.bitarray()
        remain_d = None
        syndromes = None
        if is_fix_err:
            remain_d = self.remain_dev(bit_arr)
            syndromes = make_table()

        # print("len synd", len(syndromes))
        for i in range(len(bit_arr) // self.n):
            i_beg = i * self.n
            if is_fix_err:
                # print(self.n_k)
                key = self.ba_to_str_bits(remain_d[i_beg:i_beg + self.n_k])
                index_err_bit = syndromes[key]
                if len(index_err_bit):
                    print(bit_arr)
                    for index_err in index_err_bit:
                        bit_arr[i_beg + index_err] = not bit_arr[i_beg + index_err]
                    else:
                        print(" " * (10 + i_beg + index_err_bit[0]) + "|" * (2))
                        print(bit_arr)
                        print(index_err_bit)
                        print(key)

                        print(f"{index_err_bit[0]} from {self.n}", )
                        print(f"error fount. It index is {i_beg + index_err_bit[0]}", "\n")

            decoded.extend(bit_arr[i_beg + self.n_k:i_beg + self.n])

        return decoded

    def wt(self, word):
        # Hamming weight
        np_array = self.reshape_to_np_arr(word, len(word))
        return np.count_nonzero(np_array)

    def make_table(self):
        syndromes = dict()
        for i in range(2 ** self.n):
            err = ba.bitarray(self.int_to_str_bits(i, self.n))
            if self.wt(err) <= self.t:
                # print(self.wt(err), err, len(err))
                synd = self.remain_dev(err)[:self.n_k]
                str_synd = self.ba_to_str_bits(synd)
                syndromes[str_synd] = np.nonzero(list(err))[0]
        # print(f"len(syndromes) {len(syndromes)}")
        return syndromes

    def make_table_L(self):
        syndromes = dict()
        syndromes["0" * self.n_k] = np.array([])
        for err_pac_num in range(2 ** (self.L - 2)):
            err_pac = self.get_err_pack(err_pac_num, self.L)

            for i in range(self.n - self.L + 1):
                err = ba.bitarray(self.n)
                err.setall(False)
                err[i:i + self.L] = err_pac
                synd = self.remain_dev(err)[:self.n_k]
                str_synd = self.ba_to_str_bits(synd)
                # print(str_synd, " : ", synd, " : ", err)
                syndromes[str_synd] = np.nonzero(list(err))[0]
        # print(syndromes)
        # print(f"len(syndromes) {len(syndromes)}")
        return syndromes

    @staticmethod
    def get_err_pack(decimal_num_of_num: int, L: int):
        err_pac = None
        if L > 1:
            err_pac = ba.bitarray("1" + CyclicCode.int_to_str_bits(decimal_num_of_num, L - 2) + "1")
        else:
            err_pac = ba.bitarray("1")
        return err_pac

    @staticmethod
    def ba_to_str_bits(bit_arr: ba.bitarray) -> str:
        return str(bit_arr)[10:-2]

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

        encode = self.encody_sys(bit_a)
        print(f"endcode {len(encode)}, {encode}")

        encode.tofile(file_out)

    def decode_file(self, file_in, file_out, make_table, is_fix_err: bool = True):
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
        :param make_table: function is returned the special dict. See CyclicCode.make_table()
        :param is_fix_err:
        :return:
        """
        bit_a = ba.bitarray()
        bit_a.fromfile(file_in)

        print(f"endcode {len(bit_a)}, {bit_a}")
        bit_a = self.add_to_multiplicity_n(bit_a, self.n, is_to_less=True)
        print(f"endcode {len(bit_a)}, {bit_a}")

        decode = self.decode_sys(bit_a, make_table, is_fix_err=is_fix_err)

        key = len(decode)
        true_len = self.bone.setdefault(key, None)
        if true_len:
            decode = decode[:true_len]
        # clear the memory
        del self.bone[key]

        print(f"decode  {len(decode)}, {decode}")
        # decode = self.add_to_multiplicity_n(decode, 8, is_to_less=True)
        decode.tofile(file_out)

    def make_pac_err_ba(self, bit_a: ba.bitarray):
        """

        :param bit_a: have length >= self.L
        :return:
        """
        rand_pack_num = random.randint(1, 2 ** (self.L - 2) - 1) if self.L != 2 else 0
        err_pac = self.get_err_pack(rand_pack_num, self.L)

        rand_index = random.randint(0, len(bit_a) // self.n - 1)  # выбираем блок из n бит
        rand_index = self.n * rand_index + random.randint(0, self.n - self.L)  # выбираем позицию в блоке и

        print(f"Error have started at {rand_index}")
        print(f"error pack is {err_pac}")
        print(bit_a)
        bit_a[rand_index:rand_index + self.L] = bit_a[rand_index:rand_index + self.L] ^ err_pac
        print(" " * (10 + rand_index) + "|" * self.L)
        print(bit_a)
        return bit_a

    def make_err_ba(self, bit_a: ba.bitarray):
        """

        :param bit_a: have lehght > 0
        :return:
        """
        rand_index = random.randint(0, len(bit_a) - 1)
        print(f"Create error at index {rand_index}")

        bit_a[rand_index] = not bit_a[rand_index]
        return bit_a

    def make_err_file(self, file_in, file_out, make_err_function):
        bit_a = ba.bitarray()
        bit_a.fromfile(file_in)

        bit_a = make_err_function(bit_a)
        bit_a.tofile(file_out)


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
    encode_a = main_obj.encode(bit_a)
    print(f"a: {a}")
    print(f"g: {g}")
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

    # task 4 from task list
    print("\n\nTask 4 from task list")
    endcody_a = main_obj.encody_sys(bit_a)
    print(f"endcody_a: {endcody_a}")
    print(main_obj.remain_dev(endcody_a))

    # task 5 from task list 2-3
    print("\n\nTask 5 from task list 2-3")
    print(main_obj.make_table())

    # task 5 from task list 2-3
    print("\n\nTask 6 from task list 2-3")
    with open(cur_path / "inputfile", "rb") as file_in:
        with open(cur_path / "output", "wb") as file_out:
            main_obj.encode_file(file_in, file_out)

        with open(cur_path / "output", "rb") as file_out:
            with open(cur_path / "err", "wb") as file_err:
                main_obj.make_err_file(file_out, file_err, main_obj.make_err_ba)

        with open(cur_path / "output", "wb") as file_out:
            with open(cur_path / "err", "rb") as file_err:
                main_obj.decode_file(file_err, file_out, main_obj.make_table)

    print("\n\nTask 7 from task list 2-3")
    g_new = ba.bitarray([1, 0, 0, 0, 1, 0, 1, 1, 1])
    second_obj = CyclicCode(g_new, n=15, k=7, t=2)
    print(second_obj.make_table())
    with open(cur_path / "inputfile", "rb") as file_in:
        with open(cur_path / "output", "wb") as file_out:
            second_obj.encode_file(file_in, file_out)

        with open(cur_path / "output", "rb") as file_out:
            with open(cur_path / "err", "wb") as file_err:
                second_obj.make_err_file(file_out, file_err, second_obj.make_err_ba)

        with open(cur_path / "output7", "wb") as file_out:
            with open(cur_path / "err", "rb") as file_err:
                second_obj.decode_file(file_err, file_out, second_obj.make_table)

    # Не уверен, что сделал правильно. Скорее всего есть решение с какие-нибудь сдвигами
    print("\n\nTask 8 from task list 2-3")
    gg = ba.bitarray([1, 1, 1, 1, 0, 0, 1])

    third_obj = CyclicCode(gg, n=15, k=9, L=2)
    print(third_obj.make_table_L())

    print("\n\nTask 9 from task list 2-3")
    with open(cur_path / "inputfile", "rb") as file_in:
        with open(cur_path / "output", "wb") as file_out:
            third_obj.encode_file(file_in, file_out)

        with open(cur_path / "output", "rb") as file_out:
            with open(cur_path / "err", "wb") as file_err:
                third_obj.make_err_file(file_out, file_err, third_obj.make_pac_err_ba)

        with open(cur_path / "output9", "wb") as file_out:
            with open(cur_path / "err", "rb") as file_err:
                third_obj.decode_file(file_err, file_out, third_obj.make_table_L)

    # # task 2*
    # print("\n\nTask2*")
    # identity = np.identity(7, dtype=np.int8).ravel()
    # remain_dev = main_obj.remain_dev(identity)
    # remain_dev = main_obj.reshape_to_np_arr(remain_dev, 7)
    # print(f"remain_dev: \n{remain_dev}")


if __name__ == '__main__':
    main()
