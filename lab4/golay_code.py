import numpy as np
import bitarray as ba
import pathlib
import random

from static_meth_for_encode import BaseCode

# For correct working of the path system and so that the files that are used would be in this directory
cur_path = pathlib.Path()

B_matrix = np.asarray([[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
                       [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
                       [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                       [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                       [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
                       [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
                       [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
                       [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
                       [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
                       [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.int8)
class GolayCode(BaseCode):
    def __init__(self, k: int = 12, n: int = 24, B=B_matrix):
        """
        :param k: input length msg
        :param n: encoded length msg
        :param B: matrix k x k
        """
        self.k: int = k
        self.n: int = n
        self.n_k: int = n - k
        self.B: np.ndarray = np.asarray(B)

        G = np.ndarray((k, n), dtype=np.int8)
        H = np.ndarray((n, k), dtype=np.int8)

        G[:, :k], G[:, k:] = np.eye(k), B
        H[:k, :], H[k:, :] = np.eye(k), B

        self.G = G  # G = [E | B] - in row
        self.H = H  # H = [E | B] - in column


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
        log_xor = np.matmul(bit_list, self.G) % 2
        coded_arr.extend(log_xor.ravel())
        return coded_arr

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
            # Algo from lecture
            # find syndrome s = wH
            s = np.matmul(bit_list[i], self.H) % 2
            u = None
            if GolayCode.wt(s, is_np_array=True) <= 3:
                u = np.concatenate((s, np.zeros(self.n_k, dtype=np.int8)))
            else:
                for j in range(len(self.B)):
                    log_xor = np.logical_xor(s, self.B[j])
                    if GolayCode.wt(log_xor, is_np_array=True) <= 2:
                        u = np.concatenate((log_xor, np.eye(self.n_k)[j]))
                        break

            # find syndrome b = sB
            sB = np.matmul(s, self.B) % 2
            if self.wt(sB) <= 3:
                u = np.concatenate((np.zeros(self.n_k, dtype=np.int8), sB))
            else:
                for j in range(len(self.B)):
                    log_xor = np.logical_xor(sB, self.B[j])
                    if GolayCode.wt(log_xor, is_np_array=True) <= 2:
                        u = np.concatenate((np.eye(self.n_k)[j], log_xor))
                        break
            if u is None:
                invalid_sequence = bit_arr[self.n * i:self.n * (i + 1)]
                err_str = f"""Can not decode the message {GolayCode.ba_to_str_bits(invalid_sequence)}.
Please repit the transfer of data."""
                raise ValueError(err_str)
            else:
                on_flush = np.logical_xor(bit_list[i], u)[0:self.k]
                decoded_arr.extend(on_flush)

        return decoded_arr


def test():
    test_arr = np.asarray(
        [[1, 1, 1], [1, 1, 0], [0, 1, 1]]
    )
    mat = np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 1, 0], [1, 0, 0, 0, 0, 1]])
    # print(np.logical_xor(test_arr, mat))
    print(np.matmul(test_arr, mat))
    log_xor = np.matmul(test_arr, mat) % 2
    ans = ba.bitarray()
    print(log_xor)
    # shape = log_xor.shape
    resh = log_xor.ravel()
    print(resh)
    print(ans)
    ans.extend(resh)
    print(ans)


if __name__ == '__main__':
    # test()
    main_obj = GolayCode()

    encode_1_exmp_with_err = "101" + "111" + "101" + "111" + "010" + "010" + "010" + "010"
    encode_1_exmp_with_err = ba.bitarray(encode_1_exmp_with_err)
    decode_1_exmp_with_err = main_obj.decode(encode_1_exmp_with_err)
    encode_decode_mine = main_obj.encode(decode_1_exmp_with_err)

    print(f"Take example 1 from lecture\n{encode_1_exmp_with_err}")
    print(f"Decode it\n{decode_1_exmp_with_err}")
    print(f"And encode to get correct msg\n{encode_decode_mine}")

