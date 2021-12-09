# AI 연구데이터-AI 분석활용 경진대회
## 국내 논문 문장 의미 태깅

1. 개요 
    - 연구 배경
        - 국내 논문은 주로 연구 목적, 연구 방법, 연구 결과 등으로 크게 분류되어 있습니다.
        - 제공된 데이터셋의 의미 구조 분류 태그를 **대분류** 문장 의미 세부 분류 태그를 **세부분류**로 나누었습니다.
        - 세부분류 태그의 데이터 불균형 문제를 해결하기 위해 대분류 정보를 학습에 사용한 모델을 구현하였습니다.

    - 데이터셋
        - 국내 논문 문장 의미 태깅 : [kisti](https://aida.kisti.re.kr/data/8d0fd6f4-4bf9-47ae-bd71-7d41f01ad9a6)
        - 형식
        ```json
        {
          "doc_id": "논문ID",
          "sentence": "문장 단위 텍스트",
          "tag": "문장 역할 태그",
          "keysentence": "태그별 대표 문장 여부(yes/no)"
        }
        ```
      
        <table border="1"><br>
          <thead>
            <tr>
              <th class="tg-0lax">대분류</th>
              <th class="tg-0lax">세부분류</th>
              <th class="tg-0lax">학습데이터</th>
              <th class="tg-0lax">평가데이터</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td class="tg-0lax" rowspan="3">연구 목적</td>
              <td class="tg-0lax">문제 정의</td>
              <td class="tg-0lax">14,898</td>
              <td class="tg-0lax">2,979</td>
            </tr>
            <tr>
              <td class="tg-0lax">가설 설정</td>
              <td class="tg-0lax">3,412</td>
              <td class="tg-0lax">682</td>
            </tr>
            <tr>
              <td class="tg-0lax">기술 정의</td>
              <td class="tg-0lax">12,515</td>
              <td class="tg-0lax">2,503</td>
            </tr>
            <tr>
              <td class="tg-0lax" rowspan="4">연구 방법</td>
              <td class="tg-0lax">제안 방법</td>
              <td class="tg-0lax">24,355</td>
              <td class="tg-0lax">4,871</td>
            </tr>
            <tr>
              <td class="tg-0lax">대상 데이터</td>
              <td class="tg-0lax">20,245</td>
              <td class="tg-0lax">4,049</td>
            </tr>
            <tr>
              <td class="tg-0lax">데이터처리</td>
              <td class="tg-0lax">14,234</td>
              <td class="tg-0lax">2,846</td>
            </tr>
            <tr>
              <td class="tg-0lax">이론/모형</td>
              <td class="tg-0lax">11,184</td>
              <td class="tg-0lax">2,236</td>
            </tr>
            <tr>
              <td class="tg-0lax" rowspan="2">연구 결과</td>
              <td class="tg-0lax">성능/효과</td>
              <td class="tg-0lax">37,046</td>
              <td class="tg-0lax">7,409</td>
            </tr>
            <tr>
              <td class="tg-0lax">후속연구</td>
              <td class="tg-0lax">17,850</td>
              <td class="tg-0lax">3,570</td>
            </tr>
      </table>

    - 제안 모델
        - 본 모델은 계층적 다중 레이블 임베딩을 이용한 논문 문장 수사학적 분류 모델입니다.
        - 사전 학습 언어모델과 BiLSTM, Attention, Label Embedding을 결합한 분류 모델입니다.
        - 세부분류 태그의 데이터 불균형 문제를 보완하고자 대분류와 세부분류를 동적결합하는 모델을 제안합니다.

2. 사용 방법
    - src/model/main.py 28 line
    - 원하는 mode를 설정해주고, checkpoint를 학습된 모델 파일을 참고하여 수정해야합니다.
   
    - 학습 방법
       ```python
           config = {"mode": "train",
                  "train_data_path": os.path.join(config.data_dir, "train.json"),
                  "test_data_path":  os.path.join(config.data_dir, "test.json"),
                  "analyze_data_path": os.path.join(config.data_dir, "sampling_data_5.txt"),
                  "cache_dir_path": config.cache_dir,
                  "model_dir_path": config.output_dir,
                  "checkpoint": 0,
       ```

    - 평가 방법
       ```python
           config = {"mode": "test",
                  "train_data_path": os.path.join(config.data_dir, "train.json"),
                  "test_data_path":  os.path.join(config.data_dir, "test.json"),
                  "analyze_data_path": os.path.join(config.data_dir, "sampling_data_5.txt"),
                  "cache_dir_path": config.cache_dir,
                  "model_dir_path": config.output_dir,
                  "checkpoint": N,
       ```
    
    - 데모 방법
       ```python
           config = {"mode": "demo",
                  "train_data_path": os.path.join(config.data_dir, "train.json"),
                  "test_data_path":  os.path.join(config.data_dir, "test.json"),
                  "analyze_data_path": os.path.join(config.data_dir, "sampling_data_5.txt"),
                  "cache_dir_path": config.cache_dir,
                  "model_dir_path": config.output_dir,
                  "checkpoint": N,
        ```

3. 라이브러리
    - numpy
    - konlpy
    - scikit-learn
    - scipy
    - sklearn
    - tokenizers
    - tqdm
    - transformers==4.7.0
    - urllib3
    - utils
    - torch==1.5.0
    - seqeval
    - tweepy==3.10.0
    - tensorflow-gpu==2.2.0
    - mecab-python3

4. 성능 및 효과
     - 성능
         <table border="1"><br>
           <thead>
             <tr>
               <th class="tg-0lax">사전학습모델</th>
               <th class="tg-0lax">모델</th>
               <th class="tg-0lax">대분류 acro F1</th>
               <th class="tg-0lax">대분류 micro acc</th>
               <th class="tg-0lax">세부분류 macro F1</th>
               <th class="tg-0lax">세부분류 micro acc</th>
               <th class="tg-0lax">대/세부분류 micro acc</th>
             </tr>
           </thead>
           <tbody>
             <tr>
               <td class="tg-0lax" rowspan="2">KorSciBERT</td>
               <td class="tg-0lax">LAN (세부분류)</td>
               <td class="tg-0lax">-</td>
               <td class="tg-0lax">-</td>
               <td class="tg-0lax">89.89</td>
               <td class="tg-0lax">89.66</td>
               <td class="tg-0lax">-</td>
             </tr>
             <tr>
               <td class="tg-0lax">LAN (대/세부분류)</td>
               <td class="tg-0lax">96.02</td>
               <td class="tg-0lax">95.51</td>
               <td class="tg-0lax">89.95</td>
               <td class="tg-0lax">89.81</td>
               <td class="tg-0lax">89.56</td>
             </tr>
             <tr>
               <td class="tg-0lax" rowspan="1">KLUE BERT-base</td>
               <td class="tg-0lax">LAN (대/세부분류)</td>
               <td class="tg-0lax">95.54</td>
               <td class="tg-0lax">96.05</td>
               <td class="tg-0lax">89.77</td>
               <td class="tg-0lax">89.63</td>
               <td class="tg-0lax">89.46</td>
             </tr>
             <tr>
               <td class="tg-0lax" rowspan="1">KLUE roBERTa-base</td>
               <td class="tg-0lax">LAN (대/세부분류)</td>
               <td class="tg-0lax">95.64</td>
               <td class="tg-0lax">96.16</td>
               <td class="tg-0lax">90.00</td>
               <td class="tg-0lax">95.85</td>
               <td class="tg-0lax">89.72</td>
             </tr>
       </table>
    - 효과
        - 세부분류 뿐만 아니라 대분류도 함께 예측 가능한 모델이며 대분류, 세부분류를 예측하는 모델은 macro F1 95.64%, 90%의 성능을 보입니다.

5. 대분류/세부분류 예측 결과

    ![image](https://user-images.githubusercontent.com/70934036/145163710-f82e3ed0-7f8f-4fdb-bbba-d53754978543.png)

6. 참고문헌
    - [Shang, Xichen, et al. "A Span-based Dynamic Local Attention Model for Sequential Sentence Classification." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers). 2021.](https://aclanthology.org/2021.acl-short.26/)
    - [김홍진, 김학수, "계층적 레이블 임베딩을 이용한 세부 분류 개체명 인식", 제 33회 한글 및 한국어 정보처리 학술대회 논문집, 2021](http://www.koreascience.or.kr/article/CFKO202130060679826.pdf)
    - [성수진, 김성찬, 이승우, 차정원, "문맥 정보를 이용한 논문 문장 수사학적 분류", 제 33회 한글 및 한국어 정보처리 학술대회 논문집, 2021](https://www.koreascience.or.kr/article/CFKO202130060700830.pdf)
