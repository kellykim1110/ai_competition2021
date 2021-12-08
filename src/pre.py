import json
import re
from tqdm import tqdm
import os
from collections import Counter
from sklearn.model_selection import train_test_split

from KorSciBERT.korscibert_v1_tf.tokenization_kisti import FullTokenizer

def separate_dataset(inf_dir, train_file, dev_file, test_file):
    with open(inf_dir, "r", encoding="utf-8") as inf:
        datas = json.load(inf)
    train_dataset, other = train_test_split(datas, test_size=0.2, random_state=42, shuffle=True)
    dev_dataset,  test_dataset= train_test_split(other, test_size=0.1, random_state=42, shuffle=True)

    if not os.path.isdir(train_file.split("train.json")[0]):
        os.mkdir(train_file.split("train.json")[0])

    with open(train_file, "w", encoding="utf-8") as train_f:
        json.dump(train_dataset, train_f, ensure_ascii=False, indent=4)
    train_f.close()
    with open(dev_file, "w", encoding="utf-8") as dev_f:
        json.dump(dev_dataset, dev_f, ensure_ascii=False, indent=4)
    dev_f.close()
    with open(test_file, "w", encoding="utf-8") as test_f:
        json.dump(test_dataset, test_f, ensure_ascii=False, indent=4)
    test_f.close()
    print("separate dataset!!")


def separate_datas(inf_dir):
    with open(inf_dir, "r", encoding="utf-8") as inf:
        datas = json.load(inf)
        print(datas[:1])

    first_class = "문제 정의\t가설 설정\t기술 정의".split("\t")
    second_class = "제안 방법\t대상 데이터\t데이터처리\t이론/모형".split("\t")
    third_class = "성능/효과\t후속연구".split("\t")
    all_class = []
    for c in [first_class, second_class, third_class]:
        all_class += c
        print(c)

    datas_class = [[] for _ in all_class]
    for i, data in enumerate(datas):
        for j, c in enumerate(all_class):
            if data["tag"] == c: datas_class[j].append(data)

    out_dirs = "origin_1_1 origin_1_2 origin_1_3 origin_2_1 origin_2_2 origin_2_3 origin_2_4 origin_3_1 origin_3_2".split(" ")
    print(out_dirs)
    for i, out_dir in enumerate(out_dirs):
        with open(os.path.join("../data", out_dir+".json"), 'w', encoding="utf-8") as f:
            json.dump(datas_class[i], f, ensure_ascii=False, indent=4)

    print("\n\nfinish!!")


def Change_Mecab(inf_dir, out_file, vocab_file, tokenizer_type="Mecab", k = 100):
    tokenizer = FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=False,
        tokenizer_type=tokenizer_type
    )
    with open(inf_dir, "r", encoding="utf-8") as inf:
        datas = json.load(inf)

    origin_token_dir = os.path.join("../data", tokenizer_type)
    if not os.path.isdir(origin_token_dir):
        os.mkdir(origin_token_dir)
    out_file = os.path.join(origin_token_dir,tokenizer_type+out_file+".json")

    token_list = []
    for i, data in enumerate(datas):
        #text = data["sentence"]
        text= re.sub(r'[0-9a-zA-Z]+', '', data["sentence"])
        text = re.sub('[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"\♥\♡\ㅋ\ㅠ\ㅜ\ㄱ\ㅎ\ㄲ\ㅡ]', '', text)

        tokens = tokenizer.tokenize(text)
        datas[i][tokenizer_type] = tokens
        token_list += [token for token in tokens if len(token)>1]
    topk_token_list = [word[0] for word in Counter(token_list).most_common(k)]
    if "[UNK]" in topk_token_list:
        topk_token_list = [word[0] for word in Counter(token_list).most_common(k+1) if word[0] != "[UNK]"]
    print(topk_token_list)

    for i, data in enumerate(datas):
        datas[i]["top"+str(k)] = [token for token in datas[i][tokenizer_type] if token in topk_token_list]
        del datas[i][tokenizer_type]

    with open(out_file,"w", encoding="utf-8") as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)
        f.close()


    print("\n\nfinish tokenization of "+inf_dir)



if __name__ =='__main__':
    inf_dir = "../data/origin.json"
    outf_dir = "../data"
    # separate_datas(inf_dir, outf_dir)

    train_file, dev_file, test_file = "../data/origin/train.json", "../data/origin/dev.json", "../data/origin/test.json"
    #separate_dataset(inf_dir, train_file, dev_file, test_file)

    vocab_file = "../KorSciBERT/korscibert_v1_tf/vocab_kisti.txt"

    #in_dirs = "_1_1 _1_2 _1_3 _2_1 _2_2 _2_3 _2_4 _3_1 _3_2".split(" ")
    #for in_dir in in_dirs:
    #    inf_dir = os.path.join("../data", "origin" + in_dir + ".json")
    #    Change_Mecab(inf_dir=inf_dir, out_file=in_dir, vocab_file=vocab_file, k=30)

    import pandas as pd
    datas = pd.read_csv("../bert/biaffine_model/baseline/wCNJ/test/test_result_init_weight.csv", encoding="utf8")[:-1]
    print(len(datas))
    datas = datas[datas.correct != datas.predict]
    print(len(datas))
    datas.to_csv("../bert/biaffine_model/baseline/wCNJ/test/test_result_init_weight_incorrect.csv", encoding="utf8", index=False)
    """
    # 문제 정의
    # ['연구', '에서', '고자', '으로', '한다', '따라서', '분석', '대한', '영향', '위한', '특성', '자료', '이용', '미치', '기초', '개발', '효과', '조사', '파악', '논문', '따른', '대상', '활용', '수행', '평가', '사용', '제공', '방법', '목적', '변화']
    # 차집합(difference) : ['목적', '기초', '제공', '파악', '논문', '고자']
    # 
    # 가설 설정
    # ['가설', '영향', '미칠', '가정', '에서', '으로', '실험', '정의', '##군', '연구', '유의', '보다', '차이', '프로그램', '참여', '의도', '대한', '적용', '서비스', '한다', '관계', '교육', '사용', '만족', '##감', '긍정', '점수', '행동', '조직', '정보']
    # 차집합(difference) : ['가정', '만족', '정보', '서비스', '가설', '의도', '긍정', '참여', '미칠', '##군', '점수']
    # 
    # 기술 정의
    # ['으로', '에서', '한다', '사용', '의미', '방법', '발생', '대한', '이용', '조직', '또는', '상태', '이나', '정의', '개인', '로서', '질환', '기술', '가지', '자신', '과정', '능력', '기능', '##성', '으며', '##과', '다양', '시키', '행동', '라고']
    # 차집합(difference) :['질환', '기능', '자신', '가지', '라고', '개인', '##성', '상태', '의미', '또는', '##과', '과정', '시키', '기술', '이나', '능력', '로서']
    # 
    # 제안 방법
    # ['에서', '으로', '연구', '분석', '이용', '사용', '측정', '조사', '였으며', '특성', '위해', '대한', '실험', '문항', '방법', '평가', '비교', '실시', '위하', '통해', '시간', '수행', '적용', '구성', '변화', '따른', '결과', '처리', '대상', '한다']
    # 차집합(difference) :[]
    # 
    # 대상 데이터
    # ['에서', '으로', '연구', '사용', '대상', '실험', '였으며', '분석', '조사', '자료', '##원', '부터', '까지', '이용', '##도', '대상자', '선정', '환자', '##시', '제외', '병원', '소재', '지역', '설문지', '이상', '설문', '위해', '구입', '실시', '최종']
    # 차집합(difference) :['소재', '지역', '##시', '설문', '까지', '##원', '환자', '설문지', '병원', '제외', '부터', '최종', '구입', '선정']
    # 
    # 데이터처리
    # ['분석', '이용', '실시', '검정', '특성', '으로', '위해', '통계', '였으며', '에서', '차이', '검증', '사용', '대상자', '평균', '일반', '비교', '실험', '따른', '결과', '변수', '요인', '대한', '위하', '연구', '상관', '수준', '관계', '측정', '분산']
    # 차집합(difference) :['검증', '통계', '상관', '수준', '변수', '평균', '검정', '분산', '일반']
    # 
    # 이론/모형
    # ['사용', '측정', '으로', '에서', '도구', '연구', '이용', '개발', '방법', '문항', '분석', '척도', '위해', '였으며', '수정', '평가', '대한', '따라', '적용', '위하', '시험', '한국', '보완', '모델', '구성', '##법', '수행', '조사', '##감', '##도']
    # 차집합(difference) :['모델', '시험', '보완', '수정', '척도', '한국', '##법', '도구']
    # 
    # 성능/효과
    # ['으로', '에서', '결과', '확인', '나타났', '증가', '경우', '연구', '유의', '영향', '으며', '감소', '보다', '가장', '분석', '차이', '대한', '효과', '이상', '사용', '보였', '처리', '미치', '시간', '따라', '된다', '관계', '특성', '발생', '요인']
    # 차집합(difference) :['확인', '보였', '증가', '나타났', '경우', '감소', '가장']
    # 
    # 후속연구
    # ['연구', '으로', '에서', '필요', '된다', '대한', '향후', '결과', '다양', '개발', '한다', '위한', '효과', '활용', '사료', '따라서', '적용', '분석', '추가', '교육', '판단', '통해', '대상', '추후', '프로그램', '영향', '가능', '생각', '사용', '보다']
    # 차집합(difference) :['향후', '추가', '사료', '가능', '생각', '추후', '필요', '판단']
    """



