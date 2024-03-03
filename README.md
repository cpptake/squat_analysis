# このリポジトリについて
[こちら](https://speakerdeck.com/cpptake/jia-kong-nokonpe-sukuwatutohuomupan-bie-konpenojie-fa)で発表した、スクワットから体の角度を抽出するコード  

# 環境
[こちら](https://github.com/cpptake/my_kaggle_docker)のDocker環境を利用。

# 使い方
- input/　直下にデータを抽出したい動画ファイル（.mp4）を格納  
- /scripts/movie_skelton.py 内の`CFG.mov_name`に動画ファイルの名前を記載
- `python3 scripts/movie_skelton.py`を叩く
- 詳細な分析は /notebook/find_peak_note.ipynb で実施