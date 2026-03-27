import numpy as np

def encode_matrix(input_arr):
    """
    对 shape 为 (4, 36, 36) 的二值化矩阵进行编码，返回 shape 为 (36, 36) 的编码矩阵。
    """
    code_map = {
       
    }
    #具体数据我隐藏掉了
    # 先二值化，假设原始像素0-255，阈值127，
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
