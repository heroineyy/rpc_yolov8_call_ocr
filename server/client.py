import grpc
import ocr_pb2
import ocr_pb2_grpc

_HOST = '127.0.0.1'
_PORT = '19999'


def main():
    with grpc.insecure_channel("{0}:{1}".format(_HOST, _PORT)) as channel:
        client = ocr_pb2_grpc.OcrStub(channel=channel)
        response = client.CallOcr(ocr_pb2.OcrRequest(img="1234567"))
    print("received: " + response.result)


if __name__ == '__main__':
    main()