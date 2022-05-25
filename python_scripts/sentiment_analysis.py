import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import sys


def sentiment(reply):
    lgbm_clf_save_model = joblib.load('python_scripts/sentiment_analysis.pkl')
    tfidf_save_model = joblib.load('python_scripts/tfidf.pkl')

    comment = "귀여워❤️ 언니 올해는 5월이나 종강할때 꼭 시간내봐요 ㅎㅎ선배 넘 넘 귀엽다!!..."
    content = "#하루필름"
    hashtag = "하루필름"

    if comment != "":
        comment_sentiment = list(lgbm_clf_save_model.predict(tfidf_save_model.transform([comment])))
    if content != "":
        comment_sentiment.append(lgbm_clf_save_model.predict(tfidf_save_model.transform([content])))
    if hashtag != "":
        comment_sentiment.append(lgbm_clf_save_model.predict(tfidf_save_model.transform([hashtag])))

    sentiment_result_label = int(max(comment_sentiment, key=comment_sentiment.count))

    # text = comment + " " + content + " " + hashtag
    text = reply
    music_list_df = pd.read_csv('python_scripts/music_more_list.csv', encoding="utf-8")
    # 동일 감정 label 확인
    same_label_music = music_list_df[music_list_df['label'] == sentiment_result_label][['title', 'artist', 'Lyric']]
    same_label_music.loc['crawl_data'] = ['crawling_data', 'user', text]

    # word embedding 후 유사도 측정
    tfidf_matrix = tfidf_save_model.fit_transform(same_label_music['Lyric'])
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    recommendation_need = cosine_sim[-1]

    # 첫 번째 문서와 타 문서 간 유사도가 큰 순으로 정렬한 인덱스를 추출하되 자기 자신은 제외
    sorted_index = np.argsort(recommendation_need)[::-1]
    recommend_index = sorted_index[2:11]

    recommend_music_list = same_label_music.iloc[recommend_index]

    return recommend_music_list


if __name__ == '__main__':
    content = sys.argv[1]
    res = sentiment(content)
    print(res)