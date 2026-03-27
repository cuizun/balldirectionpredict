import numpy as np

def encode_matrix(input_arr):
    """
    对 shape 为 (4, 36, 36) 的二值化矩阵进行编码，返回 shape 为 (36, 36) 的编码矩阵。
    """
    code_map = {
        '1111': -1.16915e-7,
        '0111': -8.44628e-7,
        '1011': -1.006e-6,
        '1101': -1.34093e-6,
        '0011': -2.23002e-6,
        '1110': -2.95165e-6,
        '0101': -3.19523e-6,
        '1001': -3.43273e-6,
        '0110': -4.79985e-6,
        '0001': -5.05866e-6,
        '1010': -5.91121e-6,
        '1100': -6.61761e-6,
        '0010': -7.76856e-6,
        '0100': -8.98953e-6,
        '1000': -9.571029e-6,
        '0000': -1.06794e-5,
    }
    # 先二值化，假设原始像素0-255，阈值127
    bits = (input_arr > 127).astype(int)
    # 使用np.char.add拼接字符串，避免类型不兼容
    bits_str = np.char.add(
        np.char.add(
            np.char.add(bits[0].astype(str), bits[1].astype(str)),
            bits[2].astype(str)),
        bits[3].astype(str)
    ).reshape(-1)
    flat_encoded = np.vectorize(code_map.get)(bits_str)
    encoded = flat_encoded.reshape(36, 36)
    return encoded