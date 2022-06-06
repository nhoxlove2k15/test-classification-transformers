# -*- coding: utf-8 -*-

from fastapi import FastAPI
from api_request import ThesesPredictionRequest,OneThesisPredictionRequest
from predict import *

app = FastAPI()
print(111111)
@app.get("/health_check")
async def health_check():
	return {"message": "ok"}
print(22222)

@app.post("/thesis/one-theis-predictions-1")
async def oneBookPredict(item:OneThesisPredictionRequest):
        res = {}
        test_ids = item.test_ids
        print(type(item.test_ids))
        print(test_ids)
        predictions = predict_one_thesis(test_ids)
        preds_string = PredictLabels(predictions)
        res["labels"] = preds_string
        return res
print(33333)

# @app.post("/bookgenres/books-predictions")
# async def oneBookPredict(item: BooksPredictionRequest):
# 	return item
# text =  """
#             Bảo đảm tính toàn vẹn dữ liệu bằng các dịch vụ xác thực
#             Trong xã hội thông tin ngày nay, với sự phát triển của mạng Internet cùng những giao dịch điện tử như thương mại điện tử, trao đổi các nội dung số và chính phủ điện tử thì bảo mật và an toàn thông tin đang rất được quan tâm.
#             Một số quốc gia trên thế giới đã và đang đầu tư mạnh mẽ cho việc nghiên cứu và xây dựng hạ tầng cơ sở cho dịch vụ an toàn trong viễn thông, bao gồm 4 dịch vụ cơ bản: Bí mật, xác thực, đảm bảo tính toàn vẹn thông tin và chống chối bỏ.
#             Các dịch vụ này gắn chặt với lý thuyết mật mã và an toàn thông tin. Trong bối cảnh hội cạnh tranh và hội nhập như hiện nay, ngoài việc mua và sử dụng các công nghệ tiên tiến, chúng ta cần phải tìm hiểu và xây dựng những hướng đi riêng của mình đặc biệt là trong lĩnh vực an ninh - một lĩnh vực nhạy cảm và không được tự do hoá đáp ứng cho từng ngành khác nhau,
#             Với những lý do trên, luận văn hướng tới khảo sát hệ thống an toàn thông tin cho khách hàng qua việc kiểm tra thanh toán lương tự động của khách hàng và đảm bảo an toàn qua tài khoản ATM khi rút tiền. Tiến tới mô hình lưu thông tiền tệ không dùng tiền mặt của chính phủ, đó là một trong những vấn đề trọng tâm mà ngân hàng đề ra nhằm đảm bảo an toàn dữ liệu cho khách hàng trong mọi lúc, mọi nơi. Do thời gian có hạn nên trong luận văn này em xin trình bày chương trình kiểm tra lương tự động và mô tả môđun phân hệ bảo mật máy chủ - Host Security Module (HSM) nhằm cung cấp các hàm mã hoá cần thiết để thực hiện việc mã hoá khoá, xác thực thông báo và mã hoá số PIN trong môi trường thời gian thực.
#             Trên cơ sở lý thuyết an toàn dữ liệu, và kết hợp với các thuật toán cổ điển luận văn xây dựng chương trình và mô hình ứng dụng nhằm đảm bảo tính toàn vẹn của dữ liệu bằng các dịch vụ xác thực trong hệ thống Ngân hàng Đầu tư và Phát triển Việt Nam.
#             Luận văn nghiên cứu và tìm hiểu các nội dung chính sau: Chương 1: Tổng quan về bảo mật và an toàn dữ liệu, Nghiên cứu, tìm hiểu các giải pháp đảm bảo an ninh an toàn mạng.  Nghiên cứu các công cụ và phương pháp về bảo mật. Một số các dịch vụ xác thực. Chương 2: Các hệ khoá mật mã và hàm băm. Hàm băm không khoá, Hàm băm có khoá. Chương 3: Các phương pháp đảm bảo tính toàn vẹn của dữ liệu.  Một số phương pháp cung cấp tính toàn vẹn của dữ liệu sử dụng hàm băm. Các phương pháp xác thực tỉnh nguyên bản của dữ liệu, Chương 4: Nghiên cứu xây dựng mô hình xác thực tính toàn vẹn của dữ liệu,
#             """
# a = PredictNumbers(text)
# b = PredictLabels(a)
# print(b)