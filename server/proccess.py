import os

if __name__ == '__main__':
    file_out = "python -m grpc_tools.protoc -I ./ --python_out=./ --grpc_python_out=./ ocr.proto"
    os.popen(file_out, mode='r')

# 新接口第一次需要运行这个文件
# python_out目录指定 xxxx_pb2.py的输出路径，我们指定为./ 当前路径
# grpc_python_out指定xxxx_pb2_grpc.py文件的输出路径,我们指定为./ 当前路径
# grpc_tools.protoc 这是我们的工具包，刚刚安装的
# -I参数指定协议文件的查找目录，我们都将它们设置为当前目录./
# compute.proto 我们的协议文件


# 会生成compute_pb2_grpc.py  compute_pb2.py两个文件
# compute.proto 协议文件
# compute_pb2.py 里面有消息序列化类
# compute_pb2_grpc.py 包含了服务器 Stub 类和客户端 Stub 类，以及待实现的服务 RPC 接口。