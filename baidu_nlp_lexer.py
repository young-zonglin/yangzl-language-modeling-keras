from aip import AipNlp
import parameters

def get_client():
    """
    获取百度nlp client
    :return: 百度nlp client
    """
    client = AipNlp(parameters.APP_ID, parameters.API_KEY, parameters.SECRET_KEY)
    client.setConnectionTimeoutInMillis(20000)
    client.setSocketTimeoutInMillis(60000)
    return client


def segment_baidu(client, text):
    """
    使用百度词法分析接口进行分词
    :param client: 百度nlp client
    :param text: 待分词文本
    :return: List<String>
    """
    # res: dict
    res = client.lexer(text)
    # items: list
    items = res.get('items')
    length = len(items)
    seg_list = list()
    for i in range(length):
        # item: dict
        item = items[i]
        seg_list.append(item.get('item'))
    return seg_list


text = "同意在新起点上退动中美关析取得更大法展"
client = get_client()
seg_result = segment_baidu(client, text)
for word in seg_result:
    print(word, end=" ")
