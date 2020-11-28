import numpy as np
import bitarray as ba
import random

import pathlib

# This is a sample Python script.

# For correct working of the path system and so that the files that are used would be in this directory
cur_path = pathlib.Path()



# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
def search_err(H, x):
    H_trans = np.transpose(H)
    error_index = -1
    for i in range(H_trans.shape[0]):
        print(H_trans[i])
        if all(H_trans[i][j] == x[j] for j in range(len(x))):
            error_index = i
            break

    return error_index


def main():
    # task 1
    a = [1, 0, 1, 0]

    G = [[1, 0, 0, 0, 1, 1, 1],
         [0, 1, 0, 0, 1, 1, 0],
         [0, 0, 1, 0, 1, 0, 1],
         [0, 0, 0, 1, 0, 1, 1]]

    H = [[1, 1, 1, 0, 1, 0, 0],
         [1, 1, 0, 1, 0, 1, 0],
         [1, 0, 1, 1, 0, 0, 1]]

    a = np.asarray(a)
    G = np.asanyarray(G)
    H = np.asanyarray(H)

    c = np.matmul(a, G) % 2

    print("\nTask 1")
    print(f"G:\n{G}\n\n H:\n{H}")

    print(f"shape G: {G.shape}")
    print(f"shape H: {H.shape}\n")
    print(f"checking H*G^T:\n {np.matmul(H, np.transpose(G)) % 2}\n")
    print(f"a*G:\n {c}\n")

    # task 2

    b = np.matmul(H, c) % 2
    e = np.asarray([0, 0, 0, 0, 1, 0, 0])
    c_2 = np.add(c, e) % 2
    b_2 = np.matmul(H, c_2) % 2
    H_trans = np.transpose(H)

    # поиск бита с ошибкой
    error_index = search_err(H, b_2)

    print("\nTask 2")
    print(f"check b == 0':\n {b}")
    print(f"c_2 is:\n {c_2}")
    print(f"b_2 is:\n {b_2}")
    print(f"index of error: {error_index} \nand for human: {error_index + 1}")

    # task 3

    bit_arr = ba.bitarray()
    with open(cur_path / "inputfile", "rb") as file:
        bit_arr.fromfile(file)

    print("\nTask 3")
    print(f"bit_arr: {bit_arr}")

    # task 4

    bit_list = bit_arr.tolist()
    bit_list = np.asarray(bit_list, dtype=np.int8)
    bit_list = bit_list.reshape((len(bit_list) // 4, 4))

    coded_arr = []
    # coding
    for i in range(len(bit_list)):
        coded_arr.extend(np.matmul(bit_list[i], G) % 2)

    coded_arr = np.asarray(coded_arr)
    bit_arr2 = ba.bitarray()

    with open(cur_path / "outputfile", "wb") as o_file:
        bit_arr2.extend(coded_arr)
        bit_arr2.tofile(o_file)

    print("\nTask 4")
    print(f"bit_list: {bit_list}")
    print(f"coded_arr: {coded_arr}")
    print(f"bit_arr2: {bit_arr}")

    # task 5

    bit_arr3 = ba.bitarray()
    with open(cur_path / "outputfile", "rb") as file:
        bit_arr3.fromfile(file)
    # разбиваем по массивам в 7 чисел
    bit_arr3 = bit_arr3.tolist()
    end_index = (len(bit_arr3) // 7) * 7
    bit_arr3 = bit_arr3[0: end_index]
    bit_list2 = np.asarray(bit_arr3, dtype=np.int8)
    bit_list2 = bit_list2.reshape((len(bit_list2) // 7, 7))

    # decode
    bit_decoded = ba.bitarray()
    for i in range(len(bit_list2)):
        bit_decoded.extend(bit_list2[i][:4])

    with open(cur_path / "decoded", "wb") as file:
        bit_decoded.tofile(file)

    print("\nTask 5")
    print(f"bit_decoded: {bit_decoded}")
    print(f"bit_list2: \n{bit_list2}")

    # Task 6

    bit_arc = ba.bitarray()
    with open(cur_path / "inputfile.zip", "rb") as arc:
        bit_arc.fromfile(arc)

    rand_index = random.randint(0, len(bit_arc) - 1)

    print("Task 6\n")
    print(f"index: {rand_index}")
    print(bit_arc[rand_index])
    bit_arc[rand_index] = not bit_arc[rand_index]
    print(bit_arc[rand_index])

    with open(cur_path / "inputfile_crashed.zip", "wb") as arc:
        bit_arc.tofile(arc)

    # Task 7

    print("Task 7\n")

    bit_arr3 = ba.bitarray()
    with open(cur_path / "outputfile", "rb") as file:
        bit_arr3.fromfile(file)

    # generate error
    rand_index = random.randint(0, len(bit_arr3) - 1)
    bit_arr3[rand_index] = not bit_arr3[rand_index]
    print("rand index ", rand_index)
    # we don't where is the error
    rand_index = -1

    with open(cur_path / "witherror", "wb") as file:
        bit_arr3.tofile(file)

    bit_arr3 = bit_arr3.tolist()
    end_index = (len(bit_arr3) // 7) * 7
    bit_arr3 = bit_arr3[0: end_index]
    bit_list2 = np.asarray(bit_arr3, dtype=np.int8)
    bit_list2 = bit_list2.reshape((len(bit_list2) // 7, 7))

    arr_index_err = -1
    x = None
    for i in range(len(bit_list2)):
        check_arr = np.matmul(H, bit_list2[i]) % 2
        if not all(check_arr[j] == 0 for j in range(len(check_arr))):
            arr_index_err = i
            x = check_arr
            break

    index_err = -1
    for i in range(H_trans.shape[0]):
        if all(H_trans[i][j] == x[j] for j in range(len(x))):
            index_err = i
            break

    print(f"index of 7: {arr_index_err}\nindex in 7: {index_err}")
    print(arr_index_err * 7 + index_err)

    # исправляем ошибку
    incorrect_byte_index = arr_index_err * 7 + index_err
    bit_list2[arr_index_err][index_err] = not bit_list2[arr_index_err][index_err]

    # decode
    bit_decoded = ba.bitarray()
    for i in range(len(bit_list2)):
        bit_decoded.extend(bit_list2[i][:4])

    with open(cur_path / "fixed_decoded", "wb") as file:
        temp = ba.bitarray()
        temp.extend(bit_decoded)
        temp.tofile(file)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
