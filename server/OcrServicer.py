import base64
import time
from concurrent import futures

import grpc

# 刚刚生产的两个文件
import ocr_pb2
import ocr_pb2_grpc

from pix2text import Pix2Text
import cv2
import numpy as np
from PIL import Image, ImageOps
import io

def rotate_bound_white_bg(image: object, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH), borderValue=(255, 255, 255))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


def NumNotIn(s):
    for char in s:
        if char.isdigit():
            return False
    return True

def is_contain_chinese(check_str):
    """
    判断字符串中是否包含中文
    :param check_str: {str} 需要检测的字符串
    :return: {bool} 包含返回True， 不包含返回False
    """
    for ch in check_str:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False

def ismatch(text, str):
    if str in text:
        return True
    else:
        return False


class OcrServicer(ocr_pb2_grpc.OcrServicer):
    def CallOcr(self, request, ctx):
        img_fp_bytes = request.img
        # base64解码
        str_decode = base64.b64decode(img_fp_bytes)
        np_decode = np.frombuffer(str_decode, np.uint8)
        img_fp = cv2.imdecode(np_decode, cv2.IMREAD_COLOR)
        img_fp_cop = img_fp
        img_fp1 = img_fp.astype(np.uint8)
        # 再转换成PIL Image形式
        img_fp1 = Image.fromarray(img_fp1)
        p2t = Pix2Text(analyzer_config=dict(model_name='mfd'))
        # outs = p2t(img_fp, resized_shape=600)  # 也可以使用 `p2t.recognize(img_fp)` 获得相同的结果

        outs = p2t.recognize(img_fp1)
        # print(outs)
        # 如果只需要识别出的文字和Latex表示，可以使用下面行的代码合并所有结果
        only_text = '\n'.join([out['text'] for out in outs])

        # 根据‘是否存在数字’和‘是否存在中文’作为是否翻转的条件
        # TODO 逻辑补充
        if only_text.strip() == "" or NumNotIn(only_text) or (is_contain_chinese(only_text) and NumNotIn(only_text)):
            img = cv2.imread(img_fp_cop)
            imgRotation = rotate_bound_white_bg(img, 90)
            outs = p2t.recognize(Image.fromarray(imgRotation))
            only_text = '\n'.join([out['text'] for out in outs])

        print(only_text)
        print('-----如下需要抽取有意义的数据-----')

        # matchObj = re.match(r'[-+]?[0-9]*\.?[0-9]*',only_text)
        # print(matchObj.group())

        text_list = only_text.split()
        list = []
        # print(text_list)
        for i in text_list:
            # 筛选出带数字的字符串
            if NumNotIn(i) == False:
                # 剔除掉字符串中的非数字字符
                i = "".join(filter(lambda i: i in '0123456789.-+±=X()', i))
                # 替换字符串中字符
                i = i.replace(",", ".")
                # 匹配特定字符
                if ismatch(i, "±"):
                    j = i.replace("±", "+")
                    i = str(np.round(eval(j), 3))
                if ismatch(i, "+"):
                    i = str(np.round(eval(text_list[0] + i), 3))
                if not ismatch(i, "X"):
                    list.append(i)
                if i == "1":
                    list.remove("1")
                if i == "2":
                    list.remove("2")
        num_list = np.asfarray(list, dtype=float)
        # 各层级获取最大值
        max_num = np.max(num_list)
        # 排除多数据干扰（目前只处理4种数据的情况）
        if len(num_list) == 4:
            list_sort = sorted(num_list, reverse=True)
            list_sort.remove(max_num)
            diff = list_sort[0] - list_sort[-1]
            if (max_num - list_sort[0] > np.round(diff, 3)):
                max_num = list_sort[0]
        return ocr_pb2.OcrReply(result=str(max_num))

    # def CallOcr(self, request, ctx):
    #     max_len = str(len(request.img))
    #     return ocr_pb2.OcrReply(result=max_len)


def main():
    # 多线程服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 实例化 计算len的类
    servicer = OcrServicer()
    # 注册本地服务,方法ComputeServicer只有这个是变的
    ocr_pb2_grpc.add_OcrServicer_to_server(servicer, server)
    # 监听端口
    server.add_insecure_port('127.0.0.1:19999')
    # 开始接收请求进行服务
    server.start()
    # 使用 ctrl+c 可以退出服务
    try:
        print("running...")
        # 启动后多少秒自动关闭服务，如果想保持服务进程用：server.wait_for_termination()
        time.sleep(1000)
    except KeyboardInterrupt:
        print("stopping...")
        server.stop(0)


if __name__ == '__main__':
    main()